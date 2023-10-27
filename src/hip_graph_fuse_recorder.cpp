#include "hip_graph_fuse_recorder.hpp"
#include "hip_global.hpp"
#include "hip_internal.hpp"
#include "utils/debug.hpp"
#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <yaml-cpp/yaml.h>


namespace {
void rtrim(std::string& str) { str.erase(std::find(str.begin(), str.end(), '\0'), str.end()); }

bool enabled(const std::string& value) {
  std::string lowercaseValue(value.size(), '\0');
  std::transform(value.begin(), value.end(), lowercaseValue.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  rtrim(lowercaseValue);
  static std::unordered_set<std::string> options{"1", "enable", "enabled", "yes", "true", "on"};
  if (options.find(lowercaseValue) != options.end()) {
    return true;
  }
  return false;
}

hipKernelNodeParams getKernelNodeParams(hipGraphNode* node) {
  auto* kernelNode = dynamic_cast<hipGraphKernelNode*>(const_cast<hipGraphNode*>(node));
  guarantee(kernelNode != nullptr, "failed to convert a graph node to `hipGraphKernelNode`");
  hipKernelNodeParams kernelParams;
  kernelNode->GetParams(&kernelParams);
  return kernelParams;
}

amd::Kernel* getDeviceKernel(hipKernelNodeParams& nodeParams) {
  hipFunction_t hipFunc = hipGraphKernelNode::getFunc(nodeParams, ihipGetDevice());
  auto* deviceFunc = hip::DeviceFunc::asFunction(hipFunc);
  guarantee(deviceFunc != nullptr, "failed to retrieve the kernel of a graph node");

  auto* kernel = deviceFunc->kernel();
  return kernel;
}

amd::Kernel* getDeviceKernel(hipGraphNode* node) {
  auto nodeParams = getKernelNodeParams(node);
  return getDeviceKernel(nodeParams);
}

bool equal(const dim3& one, const dim3& two) {
  return (one.x == two.x) && (one.y == two.y) && (one.z == two.z);
}

template <typename T> void append(std::vector<T>& vec) { vec.push_back(T()); };
}  // namespace


namespace hip {
bool GraphFuseRecorder::isRecordingStateQueried_{false};
bool GraphFuseRecorder::isRecordingSwitchedOn_{false};
std::string GraphFuseRecorder::tmpDirName_{};
size_t GraphFuseRecorder::imageCounter_{0};

bool GraphFuseRecorder::isInputOk() {
  auto* env = getenv("AMD_FUSION_RECORDING");
  if (env == nullptr) {
    return false;
  }
  if (!enabled(std::string(env))) {
    return false;
  }

  env = getenv("AMD_FUSION_CONFIG");
  if (env == nullptr) {
    std::stringstream msg;
    ClPrint(amd::LOG_ERROR, amd::LOG_ALWAYS,
            "fusion config is not specified; cannot proceed fusion recording");
    return false;
  }

  std::string configPathName(env);
  std::filesystem::path dirPath(configPathName);
  if (!std::filesystem::exists(dirPath)) {
    ClPrint(amd::LOG_ERROR, amd::LOG_ALWAYS, "cannot open fusion config file: %s",
            configPathName.c_str());
    return false;
  }

  try {
    auto config = YAML::LoadFile(configPathName);
    tmpDirName_ = config["tmp_dir_path"].as<std::string>();
    std::filesystem::path dirPath(tmpDirName_);
    if (!std::filesystem::exists(dirPath)) {
      auto isOk = std::filesystem::create_directories(dirPath);
      if (!isOk) {
        ClPrint(amd::LOG_ERROR, amd::LOG_ALWAYS, "failed to create a tmp dir: %s", dirPath.c_str());
        return false;
      }
    }
    std::filesystem::create_directories(dirPath);
  } catch (const YAML::ParserException& ex) {
    ClPrint(amd::LOG_ERROR, amd::LOG_ALWAYS, "error while parsing fusion config: %s", ex.what());
    return false;
  } catch (const std::runtime_error& ex) {
    ClPrint(amd::LOG_ERROR, amd::LOG_ALWAYS, "error while parsing fusion config: %s", ex.what());
    return false;
  }

  ClPrint(amd::LOG_INFO, amd::LOG_ALWAYS, "graph fuse recorder is enabled");
  return true;
}

bool GraphFuseRecorder::isRecordingOn() {
  static amd::Monitor lock_;
  amd::ScopedLock lock(lock_);
  if (!isRecordingStateQueried_) {
    isRecordingSwitchedOn_ = GraphFuseRecorder::isInputOk();
    isRecordingStateQueried_ = true;
  }
  return isRecordingSwitchedOn_;
}

void GraphFuseRecorder::run() {
  amd::ScopedLock lock(fclock_);
  const auto& nodes = graph_->GetNodes();
  if (!findCandidates(nodes)) {
    return;
  }

  std::vector<KernelImageMapType> kernelsMaps{};
  for (auto& group : fusionGroups) {
    auto map = collectImages(group);
    kernelsMaps.push_back(map);
  }
  saveFusionConfig(kernelsMaps);
}

bool GraphFuseRecorder::findCandidates(const std::vector<Node>& nodes) {
  for (size_t i = 0; i < nodes.size() - 1; ++i) {
    auto& node = nodes[i];
    const auto outDegree = node->GetOutDegree();
    if (outDegree != 1) {
      std::stringstream msg;
      msg << "cannot perform fusion because node `" << i << "` contains multiple output edges. "
          << "Number of output edges equals " << outDegree;
      ClPrint(amd::LOG_ERROR, amd::LOG_ALWAYS, msg.str().c_str());
      return false;
    }
  }

  fusionGroups.push_back(std::vector<Node>());
  fusedExecutionOrder.push_back(std::vector<size_t>());
  dim3 referenceBlockSize{};
  bool isRecording{true};
  for (size_t i = 0; i < nodes.size(); ++i) {
    auto& node = nodes[i];
    const auto type = node->GetType();
    const auto outDegree = node->GetOutDegree();
    auto& group = fusionGroups.back();
    auto& executionSequence = fusedExecutionOrder.back();

    if (type == hipGraphNodeTypeKernel) {
      isRecording = true;

      auto params = getKernelNodeParams(node);
      auto* kernel = getDeviceKernel(params);

      if (group.empty()) {
        referenceBlockSize = params.blockDim;
      }

      const bool isBlockSizeEqual = equal(referenceBlockSize, params.blockDim);
      if (isBlockSizeEqual) {
        group.push_back(node);
        executionSequence.push_back(i);
      } else {
        append(fusionGroups);
        fusionGroups.back().push_back(node);

        append(fusedExecutionOrder);
        fusedExecutionOrder.back().push_back(i);
      }
    }

    if (type != hipGraphNodeTypeKernel) {
      if (isRecording) {
        append(fusionGroups);
      }
      isRecording = false;

      append(fusedExecutionOrder);
      fusedExecutionOrder.back().push_back(i);
      continue;
    }
  }

  fusionGroups.erase(std::remove_if(fusionGroups.begin(), fusionGroups.end(),
                                    [](auto& group) { return group.size() <= 1; }),
                     fusionGroups.end());

  if (fusionGroups.empty()) {
    ClPrint(amd::LOG_ERROR, amd::LOG_ALWAYS, "could not find fusion candidates");
    return false;
  }

  size_t nodeCounter{0};
  std::for_each(fusedExecutionOrder.begin(), fusedExecutionOrder.end(),
                [&nodeCounter](auto& item) { nodeCounter += item.size(); });
  guarantee(nodeCounter == nodes.size(), "failed to process execution sequences");
  return true;
}

GraphFuseRecorder::KernelImageMapType GraphFuseRecorder::collectImages(
    const std::vector<Node>& group) {
  const auto& devices = hip::getCurrentDevice()->devices();
  const auto currentDeviceId = ihipGetDevice();
  auto device = devices[currentDeviceId];

  KernelImageMapType map{};
  for (size_t i = 0; i < group.size(); ++i) {
    const auto& node = group[i];
    auto* kernel = getDeviceKernel(node);
    auto kernelName = kernel->name();
    rtrim(kernelName);

    auto& program = kernel->program();
    auto [image, imageSize, isAllocated] = program.binary(*device);

    ImageHandle handle{};
    handle.image_ = reinterpret_cast<char*>(const_cast<uint8_t*>(image));
    handle.imageSize_ = static_cast<size_t>(imageSize);
    handle.isAllocated_ = isAllocated;

    if (imageMap.find(handle) == imageMap.end()) {
      handle.fileName = generateImagePath();
      imageMap.insert(handle);
      saveImageToDisk(handle);
    }
    auto imageName = imageMap.find(handle)->fileName;
    map.push_back({kernelName, imageName});
  }
  return map;
}

void GraphFuseRecorder::saveImageToDisk(ImageHandle& imageHandle) {
  if (imageHandle.imageSize_ > 0) {
    auto iamgeFile =
        std::fstream(imageHandle.fileName.c_str(), std::ios_base::out | std::ios_base::binary);
    if (iamgeFile) {
      iamgeFile.write(imageHandle.image_, imageHandle.imageSize_);
    } else {
      std::stringstream msg;
      msg << "failed to write image file to `" << imageHandle.fileName.c_str() << "`";
      ClPrint(amd::LOG_ERROR, amd::LOG_ALWAYS, msg.str().c_str());
    }
  }
}

void GraphFuseRecorder::saveFusionConfig(std::vector<KernelImageMapType>& kernelsMaps) {
  YAML::Emitter out;
  out << YAML::BeginMap << YAML::Key << "executionOrder" << YAML::Value << YAML::BeginSeq;
  for (auto& sequence : fusedExecutionOrder) {
    out << YAML::Flow << YAML::BeginSeq;
    for (auto& item : sequence) {
      out << item;
    }
    out << YAML::EndSeq;
  }
  out << YAML::EndSeq << YAML::EndMap;

  out << YAML::BeginMap << YAML::Key << "groups" << YAML::Value << YAML::BeginSeq;
  for (size_t id = 0; id < kernelsMaps.size(); ++id) {
    std::string groupName = std::string("group") + std::to_string(id);
    out << YAML::BeginMap << YAML::Key << groupName << YAML::Value << YAML::BeginSeq;
    for (auto& [kernelName, imageLocation] : kernelsMaps[id]) {
      out << YAML::BeginMap;
      out << YAML::Key << "name" << YAML::Value << kernelName;
      out << YAML::Key << "location" << YAML::Value << imageLocation;
      out << YAML::EndMap;
    }
    out << YAML::EndSeq;
  }
  out << YAML::EndSeq << YAML::EndMap;

  auto configPath = generateFilePath("config.yaml");
  auto configFile = std::fstream(configPath.c_str(), std::ios_base::out);
  if (configFile) {
    configFile << out.c_str() << "\n";
  } else {
    std::stringstream msg;
    msg << "failed to write yaml config file to `" << configPath.c_str() << "`";
    ClPrint(amd::LOG_ERROR, amd::LOG_ALWAYS, msg.str().c_str());
  }
}

std::string GraphFuseRecorder::generateFilePath(const std::string& name) {
  auto path = std::filesystem::path(this->tmpDirName_) / std::filesystem::path(name);
  return std::filesystem::weakly_canonical(path).string();
}

std::string GraphFuseRecorder::generateImagePath() {
  auto name = std::string("img") + std::to_string(this->imageCounter_) + std::string(".bin");
  ++(this->imageCounter_);
  return generateFilePath(name);
}
}  // namespace hip
