#pragma once

#include <cstddef>
#include <utility>
#include <vector>
#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"
#include "hip_graph_internal.hpp"
#include "platform/kernel.hpp"


namespace hip {
class GraphFuseRecorder {
  amd::Monitor fclock_{"Guards Graph Fuse-Recorder object", true};

 public:
  // GraphFuseRecorder(hipGraph_t graph) : graph_(graph) {}
  GraphFuseRecorder(hipGraph_t graph);
  void run();
  static bool isRecordingOn();

 private:
  using KernelImageMapType = std::vector<std::pair<std::string, std::string>>;
  struct ImageHandle {
    char* image_{};
    size_t imageSize_{};
    bool isAllocated_{};
    std::string fileName_{};
    bool operator==(const ImageHandle& other) const {
      return (this->image_ == other.image_) && (imageSize_ == other.imageSize_);
    }
  };
  struct ImageHash {
    template <class T> static void hashCombine(std::size_t& seed, const T& value) {
      std::hash<T> hasher;
      seed ^= hasher(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }

    size_t operator()(const ImageHandle& info) const {
      size_t result{0};
      hashCombine(result, info.image_);
      hashCombine(result, info.imageSize_);
      return result;
    }
  };

  static bool isInputOk();
  bool findCandidates(const std::vector<Node>& nodes);
  KernelImageMapType collectImages(const std::vector<Node>& nodes);
  void saveImageToDisk(ImageHandle& imageHandle);
  void saveFusionConfig(std::vector<KernelImageMapType>& kernelsMaps);
  std::string generateFilePath(const std::string& name);
  std::string generateImagePath(size_t imageId);

  hipGraph_t graph_;
  std::vector<std::vector<Node>> fusionGroups_{};
  std::vector<std::vector<size_t>> fusedExecutionOrder_{};
  size_t instanceId_{};

  static bool isRecordingStateQueried_;
  static bool isRecordingSwitchedOn_;
  static std::string tmpDirName_;
  static std::unordered_set<ImageHandle, ImageHash> imageCache_;
  static size_t instanceCounter_;
};
}  // namespace hip
