/* Copyright (c) 2019 - 2021 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#include "hip/hip_runtime.h"
#include "hip_internal.hpp"
#include "hip_platform.hpp"
#include "hip_prof_api.h"

// HIP API callback/activity

api_callbacks_table_t callbacks_table;

extern const std::string& FunctionName(const hipFunction_t f);

extern "C" {

const char* hipKernelNameRef(const hipFunction_t f) { return FunctionName(f).c_str(); }

int hipGetStreamDeviceId(hipStream_t stream) {
  if (!hip::isValid(stream)) {
    return -1;
  }
  hip::Stream* s = reinterpret_cast<hip::Stream*>(stream);
  return (s != nullptr)? s->DeviceId() : ihipGetDevice();
}

const char* hipKernelNameRefByPtr(const void* hostFunction, hipStream_t) {
  if (hostFunction == nullptr) {
    return nullptr;
  }
  return PlatformState::instance().getStatFuncName(hostFunction);
}

hipError_t hipRegisterApiCallback(uint32_t id, void* fun, void* arg) {
  return callbacks_table.set_callback(static_cast<hip_api_id_t>(id), reinterpret_cast<activity_rtapi_callback_t>(fun), arg) ?
    hipSuccess : hipErrorInvalidValue;
}

hipError_t hipRemoveApiCallback(uint32_t id) {
  return callbacks_table.set_callback(static_cast<hip_api_id_t>(id), nullptr, nullptr) ? hipSuccess : hipErrorInvalidValue;
}

hipError_t hipRegisterActivityCallback(uint32_t id, void* fun, void* arg) {
  return callbacks_table.set_activity(static_cast<hip_api_id_t>(id), reinterpret_cast<activity_sync_callback_t>(fun), arg) ?
    hipSuccess : hipErrorInvalidValue;
}

hipError_t hipRemoveActivityCallback(uint32_t id) {
  return callbacks_table.set_activity(static_cast<hip_api_id_t>(id), nullptr, nullptr) ? hipSuccess : hipErrorInvalidValue;
}

hipError_t hipEnableTracing(bool enabled) {
  callbacks_table.set_enabled(enabled);
  return hipSuccess;
}

const char* hipApiName(uint32_t id) {
  return hip_api_name(id);
}

} // extern "C"
