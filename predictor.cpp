#define _GLIBCXX_USE_CXX11_ABI 0

#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <iostream>

#include "absl/memory/memory.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#include "tensorflow/lite/profiling/profiler.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/tools/evaluation/utils.h"
#include "tensorflow/lite/string_type.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/delegates/gpu/gl_delegate.h"

#include "predictor.hpp"

#define LOG(x) std:cerr

using namespace tflite;
using std::string;

/* Pair (label, confidence) representing a prediction. */
using Prediction = std::pair<int, float>;

/*
  Predictor class takes in model file (converted into .tflite from the original .pb file
  using tflite_convert CLI tool), batch size and device mode for inference
*/
class Predictor {
  public:
    Predictor(const string &model_file, int batch, int mode);
    void Predict(float* inputData);

    std::unique_ptr<tflite::FlatBufferModel> net_;
    std::unique_ptr<tflite::Interpreter> interpreter;
    int width_, height_, channels_;
    int batch_;
    int pred_len_ = 0;
    int mode_ = 0;
    TfLiteTensor* result_;
    bool verbose = true;
    bool allow_fp16 = false;
    bool profiling = true;
    bool read_outputs = true;
};

Predictor::Predictor(const string &model_file, int batch, int mode) {
  char* model_file_char = const_cast<char*>(model_file.c_str());
  net_ = tflite::FlatBufferModel::BuildFromFile(model_file_char);
  if(!net_){
    LOG(FATAL) << "\nFailed to mmap model" << "\n";
    exit(-1);    
  }
  net_->error_reporter();
  LOG(INFO) << "resolved reporter\n";
	
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder(*net_, resolver)(&interpreter);
  if(!interpreter) {
    LOG(FATAL) << "Failed to construct interpreter\n";
  }	

  mode_ = mode;
  batch_ = batch;
  
  if(verbose) {
    LOG(INFO) << "tensors size: " << interpreter->tensors_size() << "\n";
    LOG(INFO) << "nodes size: " << interpreter->nodes__size() << "\n";
    LOG(INFO) << "inputs: " << interpreter->inputs().size() << "\n";
    LOG(INFO) << "inputs(0) name: " << interpreter->GetInputName(0) << "\n";
    int t_size = interpreter->tensors_size();
    for(int i = 0; i < t_size; i++) {
      if(interpreter->tensor(i)->name)
        LOG(INFO) << i << ": " << interpreter->tensor(i)->name << ", "
                  << interpreter->tensor(i)->bytes << ", "
                  << interpreter->tensor(i)->type << ", "
                  << interpreter->tensor(i)->params.scale << ", " 
                  << interpreter->tensor(i)->params.zero_point << "\n";
    }
  }
  interpreter->SetNumThreads(4);
}

void Predictor::Predict(float* inputData) {
  int input = interpreter->inputs()[0];
  if(verbose)
    LOG(INFO) << "input: " << input << "\n";
  const std::vector<int> inputs = interpreter->inputs();
  const std::vector<int> outputs = interpreter->outputs();
  if(verbose) {
    LOG(INFO) << "number of inputs: " << inputs.size() << "\n";
    LOG(INFO) << "number of outputs: " << outputs.size() << "\n";
  }

  switch(mode_) {
    case 1:
      #if defined(__ANDROID__)
        TfLiteGpuDelegate options = TfLiteGpuDelegateOptionsDefault();
        options.metadata = TfLiteGpuDelegateGetModelMetadata(net_->GetModel());
        if(allow_fp16)
          options.compile_options.precision_loss_allowed = 1;
        else
          options.compile_options.precision_loss_allowed = 0;
        options.compile_options.preferred_gl_object_type = TFLITE_GL_OBJECT_TYPE_FASTEST;
        options.compile_options.dynamic_batch_enabled = 0;
        auto delegate = evaluation::CreateGPUDelegate(net_, &options);
        if(!delegate) {
          LOG(INFO) << "GPU acceleration is unsupported on this platform" << "\n";
        }
        if(interpreter->ModifyGraphWithDelegate(delegate) != kTfLIteOk) {
          LOG(FATAL) << "Failed to apply " << "GPU delegate" << "\n";
        } else {
          LOG(INFO) << "Applied " < "GPU delegate" << "\n";
        }
      #else
        auto delegate = evaluation::CreateGPUDelegate(net_);
        if(!delegate) {
          LOG(INFO) << "GPU acceleraton is unsupported on this platform" << "\n";
        }
        if(interpreter->ModifyGraphWithDelegate(delegate) != kTfLIteOk) {
          LOG(FATAL) << "Failed to apply " << "GPU delegate" << "\n";
        } else {
          LOG(INFO) << "Applied " < "GPU delegate" << "\n";
        }
      break;
    case 2:
      auto delegate = evaluation::CreateNNAPIDelegate();
      if(!delegate) {
        LOG(INFO) << "NNAPI acceleration is unsupported on this platform" << "\n";
      }
      interpreter->UseNNAPI(true);
      break;
    default:
      interpreter->SetNumThreads(4);
  }
  
  if(interpreter->AllocateTensors() != kTfLiteOk) {
    LOG(FATAL) << "Failed to allocate tensors!";
  }

  // fill input buffers
  TfLiteTensor* input_tensor = interpreter->tensor(input);
  TfLiteIntArray* input_dims = input_tensor->dims;
  height_ = input_dims->data[1];
  width_ = input_dims->data[2];
  channels_ = input_dims->data[3];

  assert(input_dims->size == 4);
  if(interpreter->tensor(input)->type != kTfLiteFloat32)
    LOG(FATAL) << "Only support Float32 models as of now" << "\n";

  const int size = batch_ * width_ * height_ * channels_;
  memcpy(input_tensor->data.f, &inputData[0], size);

  const int output = interpreter->outputs()[0];
  result_ = interpreter->tensor(output);

  auto profiler = absl::make_unique<profiling::Profiler>(1024);
  interpreter->SetProfiler(profiler.get());
  if(profiling)
    profiler->StartProfiling();
  
  // run inference
  if(interpreter->Invoke() != kTfLiteOk) {
    LOG(FATAL) << "Failed to invoke tflite" << "\n";
  }

  if(profiling) {
    profiler->StopProfiling();
    auto profile_events = profiler->GetProfilerEvents();
    for(int i = 0; i < profiler_events.size(); i++) {
      auto op_index = profiler_events[i]->event_metadata;
      const auto node_and_registration = interpreter->node_and_registration(op_index);
      const TfLiteRegistration registration = node_and_registration->second;
      LOG(INFO) << std::fixed << std::setw(10) << std::setprecision(3)
                << (profile_events[i]->end_timestamp_us - profile_events[i]->begin_timestamp_us) / 1000.0
                << ", Node" << std::setw(3) << std::setprecision(3) << op_index
                << ", OpCode" << std::setw(3) << std::setprecision(3)
                << registration.builtin_code << ", "
                << EnumNameBuiltinOperator(static_cast<BuiltinOperator>(registration.builtin_code))
                << "\n";
    }

  }

  // Note: TfLiteTensor does not provide a size() API call which means we have to fetch the number of bytes the tensor
  // has and divide it by 4 since we assume float is 4 bytes long
  // Potential Bug location
  //pred_len_ = result_->bytes/(4*batch_);
  assert(result_->dims->size == 2);
  assert(result_->dims->data[0] == 1);
  pred_len_ = result_->dims->data[1];

  if(result_->type != kTfLiteFloat32) {
    LOG(FATAL) << "Expected a Float32 output" << "\n";
  }
	
  if(result_->data.f == nullptr) {
    LOG(FATAL) << "Got a NULL output" << "\n";
  }
}

PredictorContext NewTflite(char *model_file, int batch, int mode) {
  try {
    const auto ctx = new Predictor(model_file, batch, mode);
    return (void *) ctx;
  } catch(const std::invalid_argument &ex) {
    errno = EINVAL;
    return nullptr;
  }
}

void InitTflite() {}

void PredictTflite(PredictorContext pred, float* inputData) {
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return;
  }
  predictor->Predict(inputData);
  return;
}

float* GetPredictionsTflite(PredictorContext pred) {
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return nullptr;
  }
  if(predictor->result_ == nullptr) {
    throw std::runtime_error("expected a non-nil result");	
  }
  if(!(predictor->result_->type == kTfLiteFloat32)) {
    throw std::runtime_error("reuslt_->type is not Float32");
  }
  if(predictor->result_->data.f == nullptr) {
    throw std::runtime_error("expected a non-nil result->data.f");
  }

  if(predictor->result_->type == kTfLiteFloat32) {
    return predictor->result_->data.f;
  }else{
    return nullptr;
  }
}

void DeleteTflite(PredictorContext pred) {
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return;
  }
  delete predictor;
}

int GetWidthTflite(PredictorContext pred) {
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return 0;
  }
  return predictor->width_;
}

int GetHeightTflite(PredictorContext pred) {
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
  return 0;
  }
  return predictor->height_;
}

int GetChannelsTflite(PredictorContext pred) {
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return 0;
  }
  return predictor->channels_;
}

int GetPredLenTflite(PredictorContext pred) {
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return 0;
  }
  return predictor->pred_len_;
}
