#include <onnxruntime_cxx_api.h>

#include <array>
#include <filesystem>
#include <memory>
#include <string>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#define VOICEVOX_CORE_EXPORTS
#include "core.h"

#define NOT_INITIALIZED_ERR "Call initialize() first."
#define NOT_FOUND_ERR "No such file or directory: "
#define ONNX_ERR "ONNX raise exception: "
#define CUDA_IS_NOT_SUPPORTED_ERR "This library is CPU version. use_gpu is not supported."

namespace fs = std::filesystem;
constexpr std::array<int64_t, 0> scalar_shape{};
constexpr std::array<int64_t, 1> speaker_shape{1};

struct Status {
  Status(const char *root_dir_path_utf8, bool use_gpu_)
      : root_dir_path(root_dir_path_utf8),
        use_gpu(use_gpu_),
        cpu_info(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU)),
        gpu_info(nullptr) {
    // deprecated in C++20; Use char8_t for utf-8 char in the future.
    fs::path root = fs::u8path(root_dir_path);
    Ort::SessionOptions session_options;
#ifdef USE_CUDA
    if (use_gpu) {
      Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0));
      gpu_info = Ort::MemoryInfo("Cuda", OrtArenaAllocator, 0, OrtMemTypeDefault);
    }
#endif
    yukarin_s = std::make_shared<Ort::Session>(env, (root / "yukarin_s.onnx").c_str(), session_options);
    yukarin_sa = std::make_shared<Ort::Session>(env, (root / "yukarin_sa.onnx").c_str(), session_options);
    decode = std::make_shared<Ort::Session>(env, (root / "decode.onnx").c_str(), session_options);
  }

  std::string root_dir_path;
  bool use_gpu;
  Ort::MemoryInfo cpu_info, gpu_info;

  Ort::Env env{ORT_LOGGING_LEVEL_ERROR};
  std::shared_ptr<Ort::Session> yukarin_s, yukarin_sa, decode;
};

static std::string error_message;
static bool initialized = false;
static std::unique_ptr<Status> status;

template <typename T, size_t Rank>
Ort::Value ToTensor(T *data, const std::array<int64_t, Rank> &shape) {
  int64_t count = 1;
  for (int64_t dim : shape) {
    count *= dim;
  }
  return Ort::Value::CreateTensor<T>(status->cpu_info, data, count, shape.data(), shape.size());
}

bool yukarin_s_forward_cpu(int64_t length, int64_t *phoneme_list, int64_t *speaker_id, float *output) {
  try {
    const char *inputs[] = {"phoneme_list", "speaker_id"};
    const char *outputs[] = {"phoneme_length"};
    const std::array<int64_t, 1> phoneme_shape{length};

    std::array<Ort::Value, 2> input_tensors = {ToTensor(phoneme_list, phoneme_shape),
                                               ToTensor(speaker_id, speaker_shape)};
    Ort::Value output_tensor = ToTensor(output, phoneme_shape);

    status->yukarin_s->Run(Ort::RunOptions{nullptr}, inputs, input_tensors.data(), input_tensors.size(), outputs,
                           &output_tensor, 1);
  } catch (const Ort::Exception &e) {
    error_message = ONNX_ERR;
    error_message += e.what();
    return false;
  }

  return true;
}

bool yukarin_sa_forward_cpu(int64_t length, int64_t *vowel_phoneme_list, int64_t *consonant_phoneme_list,
                            int64_t *start_accent_list, int64_t *end_accent_list, int64_t *start_accent_phrase_list,
                            int64_t *end_accent_phrase_list, int64_t *speaker_id, float *output) {
  try {
    const char *inputs[] = {
        "length",          "vowel_phoneme_list",       "consonant_phoneme_list", "start_accent_list",
        "end_accent_list", "start_accent_phrase_list", "end_accent_phrase_list", "speaker_id"};
    const char *outputs[] = {"f0_list"};
    const std::array<int64_t, 1> phoneme_shape{length};

    std::array<Ort::Value, 8> input_tensors = {ToTensor(&length, scalar_shape),
                                               ToTensor(vowel_phoneme_list, phoneme_shape),
                                               ToTensor(consonant_phoneme_list, phoneme_shape),
                                               ToTensor(start_accent_list, phoneme_shape),
                                               ToTensor(end_accent_list, phoneme_shape),
                                               ToTensor(start_accent_phrase_list, phoneme_shape),
                                               ToTensor(end_accent_phrase_list, phoneme_shape),
                                               ToTensor(speaker_id, speaker_shape)};
    Ort::Value output_tensor = ToTensor(output, phoneme_shape);

    status->yukarin_sa->Run(Ort::RunOptions{nullptr}, inputs, input_tensors.data(), input_tensors.size(), outputs,
                            &output_tensor, 1);
  } catch (const Ort::Exception &e) {
    error_message = ONNX_ERR;
    error_message += e.what();
    return false;
  }
  return true;
}


bool decode_forward_cpu(int length, int phoneme_size, float *f0, float *phoneme, int64_t *speaker_id, float *output) {
  try {
    const char *inputs[] = {"f0", "phoneme", "speaker_id"};
    const char *outputs[] = {"wave"};
    const std::array<int64_t, 1> wave_shape{length * 256};
    const std::array<int64_t, 2> f0_shape{length, 1}, phoneme_shape{length, phoneme_size};

    std::array<Ort::Value, 3> input_tensor = {ToTensor(f0, f0_shape), ToTensor(phoneme, phoneme_shape),
                                              ToTensor(speaker_id, speaker_shape)};
    Ort::Value output_tensor = ToTensor(output, wave_shape);

    status->decode->Run(Ort::RunOptions{nullptr}, inputs, input_tensor.data(), input_tensor.size(), outputs,
                        &output_tensor, 1);
  } catch (const Ort::Exception &e) {
    error_message = ONNX_ERR;
    error_message += e.what();
    return false;
  }

  return true;
}

#ifdef USE_CUDA
template <typename T, size_t Rank>
Ort::Value ToGPUTensor(T *data, const std::array<int64_t, Rank> &shape) {
  int64_t count = 1;
  for (int64_t dim : shape) {
    count *= dim;
  }
  return Ort::Value::CreateTensor<T>(status->gpu_info, data, count, shape.data(), shape.size());
}

template <typename T>
Ort::MemoryAllocation ToGPU(Ort::Allocator &alloc, T *data, size_t count) {
  auto cdata = alloc.GetAllocation(sizeof(T) * count);
  cudaMemcpy(cdata.get(), data, cdata.size(), cudaMemcpyHostToDevice);
  return cdata;
}

bool yukarin_s_forward_gpu(int64_t length, int64_t *phoneme_list, int64_t *speaker_id, float *output) {
  try {
    const char *inputs[] = {"phoneme_list", "speaker_id"};
    const char *outputs[] = {"phoneme_length"};
    const std::array<int64_t, 1> phoneme_shape{length};

    Ort::Allocator cuda_allocator(status->yukarin_s, status->gpu_info);
    auto cuda_phoneme_list = ToGPU(cuda_allocator, phoneme_list, length);
    auto cuda_speaker_id = ToGPU(cuda_allocator, speaker_id, 1);
    auto cuda_output = ToGPU(cuda_allocator, output, length);

    std::array<Ort::Value, 2> input_tensors = {ToGPUTensor<int64_t>(cuda_phoneme_list.get(), phoneme_shape),
                                               ToGPUTensor<int64_t>(cuda_speaker_id.get(), speaker_shape)};
    Ort::Value output_tensor = ToGPUTensor<float>(cuda_output.get(), cuda_phoneme_shape);

    status->yukarin_s->Run(Ort::RunOptions{nullptr}, inputs, input_tensors.data(), input_tensors.size(), outputs,
                           &output_tensor, 1);
    cudaMemcpy(output, cuda_output.get(), cuda_output.size(), cudaMemcpyDeviceToHost);
  } catch (const Ort::Exception &e) {
    error_message = ONNX_ERR;
    error_message += e.what();
    return false;
  }
  return true;
}

bool yukarin_sa_forward_gpu(int64_t length, int64_t *vowel_phoneme_list, int64_t *consonant_phoneme_list,
                            int64_t *start_accent_list, int64_t *end_accent_list, int64_t *start_accent_phrase_list,
                            int64_t *end_accent_phrase_list, int64_t *speaker_id, float *output) {
  return false;
}

bool decode_forward_gpu(int length, int phoneme_size, float *f0, float *phoneme, int64_t *speaker_id, float *output) {
  try {
    const char *inputs[] = {"f0", "phoneme", "speaker_id"};
    const char *outputs[] = {"wave"};
    const std::array<int64_t, 1> wave_shape{length * 256};
    const std::array<int64_t, 2> f0_shape{length, 1}, phoneme_shape{length, phoneme_size};

    Ort::Allocator cuda_allocator(status->decode, status->gpu_info);
    auto cuda_f0 = ToGPU(cuda_allocator, f0, length);
    auto cuda_phoneme = ToGPU(cuda_allocator, phoneme, length);
    auto cuda_speaker_id = ToGPU(cuda_allocator, speaker_id, 1);
    auto cuda_output = ToGPU(cuda_allocator, output, length * 256);

    std::array<Ort::Value, 3> input_tensor = {ToGPUTensor<float>(cuda_f0.get(), f0_shape),
                                              ToGPUTensor<float>(cuda_phoneme.get(), phoneme_shape),
                                              ToGPUTensor<int64_t>(cuda_speaker_id.get(), speaker_shape)};
    Ort::Value output_tensor = ToGPUTensor(cuda_output.get(), wave_shape);

    status->decode->Run(Ort::RunOptions{nullptr}, inputs, input_tensor.data(), input_tensor.size(), outputs,
                        &output_tensor, 1);
    cudaMemcpy(output, cuda_output.get(), cuda_output.size(), cudaMemcpyDeviceToHost);
  } catch (const Ort::Exception &e) {
    error_message = ONNX_ERR;
    error_message += e.what();
    return false;
  }
  return true;
}
#endif

bool initialize(const char *root_dir_path, bool use_gpu) {
  initialized = false;
  try {
    status = std::make_unique<Status>(root_dir_path, use_gpu);
  } catch (const Ort::Exception &e) {
    error_message = ONNX_ERR;
    error_message += e.what();
    return false;
  }

  initialized = true;
  return true;
}

// TODO: 未実装
const char *metas() { return ""; }

bool yukarin_s_forward(int length, long *phoneme_list, long *speaker_id, float *output) {
  if (!initialized) {
    error_message = NOT_INITIALIZED_ERR;
    return false;
  }
  int64_t speaker_id_ll = static_cast<int64_t>(*speaker_id);
  if (status->use_gpu) {
#ifdef USE_CUDA
    return yukarin_s_forward_gpu(length, (int64_t *)phoneme_list, &speaker_id_ll, output);
#else
    error_message = CUDA_IS_NOT_SUPPORTED_ERR;
    return false;
#endif
  } else {
    return yukarin_s_forward_cpu(length, (int64_t *)phoneme_list, &speaker_id_ll, output);
  }
}

bool yukarin_sa_forward(int length, long *vowel_phoneme_list, long *consonant_phoneme_list, long *start_accent_list,
                        long *end_accent_list, long *start_accent_phrase_list, long *end_accent_phrase_list,
                        long *speaker_id, float *output) {
  if (!initialized) {
    error_message = NOT_INITIALIZED_ERR;
    return false;
  }
  int64_t speaker_id_ll = static_cast<int64_t>(*speaker_id);
  if (status->use_gpu) {
#ifdef USE_CUDA
    return yukarin_sa_forward_gpu(length, (int64_t *)vowel_phoneme_list, (int64_t *)consonant_phoneme_list,
                                  (int64_t *)start_accent_list, (int64_t *)end_accent_list,
                                  (int64_t *)start_accent_phrase_list, (int64_t *)end_accent_phrase_list,
                                  &speaker_id_ll, output);
#else
    error_message = CUDA_IS_NOT_SUPPORTED_ERR;
    return false;
#endif
  } else {
    return yukarin_sa_forward_cpu(length, (int64_t *)vowel_phoneme_list, (int64_t *)consonant_phoneme_list,
                                  (int64_t *)start_accent_list, (int64_t *)end_accent_list,
                                  (int64_t *)start_accent_phrase_list, (int64_t *)end_accent_phrase_list,
                                  &speaker_id_ll, output);
  }
}

bool decode_forward(int length, int phoneme_size, float *f0, float *phoneme, long *speaker_id, float *output) {
  if (!initialized) {
    error_message = NOT_INITIALIZED_ERR;
    return false;
  }
  int64_t speaker_id_ll = static_cast<int64_t>(*speaker_id);
  if (status->use_gpu) {
#ifdef USE_CUDA
    return decode_forward_gpu(length, phoneme_size, f0, phoneme, &speaker_id_ll, output);
#else
    error_message = CUDA_IS_NOT_SUPPORTED_ERR;
    return false;
#endif
  } else {
    return decode_forward_cpu(length, phoneme_size, f0, phoneme, &speaker_id_ll, output);
  }
}

const char *last_error_message() { return error_message.c_str(); }