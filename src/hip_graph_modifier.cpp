#include "hip_graph_modifier.hpp"
#include "hip_graph_fuse_recorder.hpp"
#include <cstddef>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <utility>
#include <vector>
#include "utils/debug.hpp"
#include <yaml-cpp/yaml.h>


namespace {
void loadExternalSymbols(const std::vector<std::pair<std::string, std::string>>& fusedGroups) {
  HIP_INIT_VOID();
  for (auto& [symbolName, imagePath]: fusedGroups) {
    PlatformState::instance().loadExternalSymbol(symbolName, imagePath);
  }
}

dim3 max(dim3& one, dim3& two) {
  return dim3(std::max(one.x, two.x), std::max(one.y, two.y), std::max(one.z, two.z));
}
}


namespace hip {
bool GraphModifier::isSubstitutionStateQueried_{false};
bool GraphModifier::isSubstitutionSwitchedOn_{false};
std::unordered_map<std::string, amd::Kernel*> GraphModifier::symbolTable_{};
std::vector<std::vector<size_t>> GraphModifier::executionOrder_{};

bool GraphModifier::isInputOk() {
  auto* env = getenv("AMD_FUSION_MANIFEST");
  if (env == nullptr) {
    std::stringstream msg;
    ClPrint(amd::LOG_ERROR, amd::LOG_ALWAYS,
            "fusion manifest is not specified; cannot proceed fusion substitution");
    return false;
  }

  std::string manifestPathName(env);
  std::filesystem::path filePath(manifestPathName);
  if (!std::filesystem::exists(filePath)) {
    ClPrint(amd::LOG_ERROR, amd::LOG_ALWAYS, "cannot open fusion manifest file: %s",
            manifestPathName.c_str());
    return false;
  }

  std::vector<std::pair<std::string, std::string>> fusedGroups{};
  std::vector<std::vector<size_t>> executionOrder{};
  try {
    auto manifest = YAML::LoadFile(manifestPathName);
    for (auto group: manifest["fusedGroups"]) {
      auto name = group["name"].as<std::string>();
      auto location = group["location"].as<std::string>();
      fusedGroups.push_back(std::make_pair(name, location));
    }
    for (auto sequence: manifest["executionOrder"]) {
      executionOrder.push_back(sequence.as<std::vector<size_t>>());
    }
  } catch (const YAML::ParserException& ex) {
    ClPrint(amd::LOG_ERROR, amd::LOG_ALWAYS, "error while parsing fusion manifest: %s", ex.what());
    return false;
  } catch (const std::runtime_error& ex) {
    ClPrint(amd::LOG_ERROR, amd::LOG_ALWAYS, "error while parsing fusion manifest: %s", ex.what());
    return false;
  }

  loadExternalSymbols(fusedGroups);
  GraphModifier::symbolTable_ = PlatformState::instance().getExternalSymbolTable();
  GraphModifier::executionOrder_ = executionOrder;
  PlatformState::instance().initSemaphore();

  ClPrint(amd::LOG_INFO, amd::LOG_ALWAYS, "graph fuse substitution is enabled");
  return true;
}

bool GraphModifier::isSubstitutionOn() {
  static amd::Monitor lock_;
  amd::ScopedLock lock(lock_);
  if (!isSubstitutionStateQueried_) {
    isSubstitutionSwitchedOn_ = GraphModifier::isInputOk();
    isSubstitutionStateQueried_ = true;
  }
  return isSubstitutionSwitchedOn_;
}

void GraphModifier::run() {
  amd::ScopedLock lock(fclock_);
  // TODO: modify the graph by embedding fused kernels
}
} // namespace hip
