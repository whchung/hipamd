#pragma once

#include <cstddef>
#include <unordered_map>
#include <utility>
#include <vector>
#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"
#include "hip_graph_internal.hpp"
#include "platform/kernel.hpp"


namespace hip {
class GraphModifier {
  amd::Monitor fclock_{"Guards Graph Modifier object", true};

 public:
  GraphModifier(hipGraph_t& graph) : graph_(graph) {};
  void run();
  static bool isSubstitutionOn();

  private:
  static bool isInputOk();

  hipGraph_t& graph_;

  static bool isSubstitutionStateQueried_;
  static bool isSubstitutionSwitchedOn_;
  static std::unordered_map<std::string, amd::Kernel*> symbolTable_;
  static std::vector<std::vector<size_t>> executionOrder_;
};
}  // namespace hip
