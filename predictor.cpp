#define _GLIBCXX_USE_CXX11_ABI 0

#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <iostream>
#include <iomanip>
#include <sys/time.h>

#include "absl/memory/memory.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#include "tensorflow/lite/profiling/profiler.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/tools/evaluation/utils.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/delegates/gpu/gl_delegate.h"

#include "predictor.hpp"

#define LOG(x) std::cerr

using namespace tflite;
using std::string;

double get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

/*
  Predictor class takes in model file (converted into .tflite from the original .pb file
  using tflite_convert CLI tool), batch size and device mode for inference
*/
class Predictor {
  public:
    Predictor(const string &model_file, int batch, int mode, bool verbose, bool profile);
    void Predict(int* inputData_quantize, float* inputData_float, bool quantize);

    std::unique_ptr<tflite::FlatBufferModel> net_;
    std::unique_ptr<tflite::Interpreter> interpreter;
    int width_, height_, channels_;
    int batch_;
    int pred_len_ = 0;
    int mode_ = 0;
    TfLiteTensor* result_;
    float* result_float_;
    bool quantize_ = false;
    bool verbose_ = false; // display model details
    bool allow_fp16_ = false;
    bool profile_ = true; // operator level profiling
    bool read_outputs_ = true;
};

Predictor::Predictor(const string &model_file, int batch, int mode, bool verbose, bool profile) {
  char* model_file_char = const_cast<char*>(model_file.c_str());
  
  profile_ = profile;
  verbose_ = verbose;
 
  // profile model loading
  struct timeval start_time, stop_time;
  gettimeofday(&start_time, nullptr); 
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
  gettimeofday(&stop_time, nullptr);
  if(verbose) {
    LOG(INFO) << "Model loading (C++): " << (get_us(stop_time) - get_us(start_time))/1000 << "ms \n";
  }
  mode_ = mode;
  batch_ = batch;
  
  if(verbose_) {
    LOG(INFO) << "tensors size: " << interpreter->tensors_size() << "\n";
    LOG(INFO) << "nodes size: " << interpreter->nodes_size() << "\n";
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
}

void Predictor::Predict(int* inputData_quantize, float* inputData_float, bool quantize) {
  int input = interpreter->inputs()[0];
  if(verbose_)
    LOG(INFO) << "input: " << input << "\n";
  const std::vector<int> inputs = interpreter->inputs();
  const std::vector<int> outputs = interpreter->outputs();
  if(verbose_) {
    LOG(INFO) << "number of inputs: " << inputs.size() << "\n";
    LOG(INFO) << "number of outputs: " << outputs.size() << "\n";
  }

  switch(mode_) {
    case 7: {
      const TfLiteGpuDelegateOptions options = {
        .metadata = NULL,
        .compile_options = {
          .precision_loss_allowed = 1, // FP16
          .preferred_gl_object_type = TFLITE_GL_OBJECT_TYPE_FASTEST,
          .dynamic_batch_enabled = 0, // Not fully functional yet
        },
      };
      auto* delegate = TfLiteGpuDelegateCreate(&options);
      if(!delegate) {
        LOG(FATAL) << "Unable to create GPU delegate" << "\n";
      }
      if(interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk) {
         LOG(FATAL) << "Failed to apply " << "GPU delegate" << "\n";
      } else {
         LOG(INFO) << "Applied " << "GPU delegate" << "\n";
      }
      break; }
    case 8: {
      auto delegate = tflite::evaluation::CreateNNAPIDelegate();
      if(!delegate) {
        LOG(INFO) << "NNAPI acceleration is unsupported on this platform" << "\n";
      }
      interpreter->UseNNAPI(true);
      break; }
    case 1: {
      interpreter->SetNumThreads(1); 
      break; }
    case 2: {
      interpreter->SetNumThreads(2); 
      break; }
    case 3: {
      interpreter->SetNumThreads(3); 
      break; }
    case 4: {
      interpreter->SetNumThreads(4); 
      break; }
    case 5: {
      interpreter->SetNumThreads(5); 
      break; }
    case 6: {
      interpreter->SetNumThreads(6);
      break; }
    default: {
      interpreter->SetNumThreads(4); }
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
  if(verbose_) {
    LOG(INFO) << "Model input height is " << height_ << "\n";
    LOG(INFO) << "Model input width is " << width_ << "\n";
    LOG(INFO) << "Model input channel is " << channels_ << "\n";
  }
  assert(input_dims->size == 4);

  const int size = batch_ * width_ * height_ * channels_;
  if(interpreter->tensor(input)->type == kTfLiteFloat32 && quantize == false) {
    LOG(INFO) << "Running float model" << "\n";
    memcpy(interpreter->typed_tensor<float>(input), &inputData_float[0], size);
  } else if (interpreter->tensor(input)->type == kTfLiteUInt8 && quantize == true) {
    LOG(INFO) << "Running quantized model" << "\n";
    uint8_t* base_pointer = interpreter->typed_tensor<uint8_t>(input);
    for(int i = 0; i < size; i++) {
      base_pointer[i] = (uint8_t)inputData_quantize[i];
    }
  } else {
    LOG(FATAL) << "Unsupported input type: " << interpreter->tensor(input)->type << "\n";
  }

  //const int output = interpreter->outputs()[0];
  //result_ = interpreter->tensor(output);

  auto profiler = absl::make_unique<profiling::Profiler>(1024);
  interpreter->SetProfiler(profiler.get());
  if(profile_ == true) {
    LOG(INFO) << "Starting profiler" << "\n";
    profiler->StartProfiling();
  }
  struct timeval start_time, stop_time;
  gettimeofday(&start_time, nullptr);  
  // run inference
  if(interpreter->Invoke() != kTfLiteOk) {
    LOG(FATAL) << "Failed to invoke tflite" << "\n";
  }
  gettimeofday(&stop_time, nullptr);
  if(verbose_) {
    LOG(INFO) << "Model computation (C++): " << (get_us(stop_time) - get_us(start_time))/1000 << "ms \n"; 
  }

  if(profile_ == true) {
    LOG(INFO) << "Stopping profiler" << "\n";
    profiler->StopProfiling();
    auto profile_events = profiler->GetProfileEvents();
    for(int i = 0; i < profile_events.size(); i++) {
      LOG(INFO) << "Inside profiler loop" << "\n";
      auto op_index = profile_events[i]->event_metadata;
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
    LOG(INFO) << "Displayed layer wise profiling information" << "\n" ;
  }

  // Note: TfLiteTensor does not provide a size() API call which means we have to fetch the number of bytes the tensor
  // has and divide it by 4 since we assume float is 4 bytes long
  // Potential Bug location
  //pred_len_ = result_->bytes/(4*batch_);
  //assert(result_->dims->size == 2);
  //assert(result_->dims->data[0] == 1);
  //pred_len_ = result_->dims->data[1];

  //if(result_->type != kTfLiteFloat32) {
  //  LOG(FATAL) << "Expected a Float32 output" << "\n";
  //}

  int output = interpreter->outputs()[0];
  TfLiteIntArray* output_dims = interpreter->tensor(output)->dims;
  auto output_size = output_dims->data[output_dims->size-1];
  pred_len_ = output_size;
  
  result_float_ = new float[output_size];
  if(interpreter->tensor(output)->type == kTfLiteFloat32) {
    float* prediction = interpreter->typed_output_tensor<float>(0);
    for(int i = 0; i < output_size; i++)
      result_float_[i] = prediction[i]; 
  }	else if(interpreter->tensor(output)->type == kTfLiteUInt8) {
    uint8_t* prediction = interpreter->typed_output_tensor<uint8_t>(0);
    for(int i = 0; i < output_size; i++)
      result_float_[i] = prediction[i] / 255.0; 
  } else {
    LOG(FATAL) << "Unsupported output type: " << interpreter->tensor(output)->type << "\n";
  }
  
  quantize_ = quantize;

  //if(result_->data.f == nullptr) {
  //  LOG(FATAL) << "Got a NULL output" << "\n";
  //}

}

PredictorContext NewTflite(char *model_file, int batch, int mode, bool verbose, bool profile) {
  try {
    const auto ctx = new Predictor(model_file, batch, mode, verbose, profile);
    return (void *) ctx;
  } catch(const std::invalid_argument &ex) {
    errno = EINVAL;
    return nullptr;
  }
}

void InitTflite() {}

void PredictTflite(PredictorContext pred, int* inputData_quantize, float* inputData_float, bool quantize) {
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return;
  }
  predictor->Predict(inputData_quantize, inputData_float, quantize);
  return;
}

float* GetPredictionsTflite(PredictorContext pred) {
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return nullptr;
  }
  //if(predictor->result_ == nullptr) {
  //  throw std::runtime_error("expected a non-nil result");	
  //}
  //if(!(predictor->result_->type == kTfLiteFloat32)) {
  //  throw std::runtime_error("reuslt_->type is not Float32");
  //}
  //if(predictor->result_->data.f == nullptr) {
  //  throw std::runtime_error("expected a non-nil result->data.f");
  //}

  //if(predictor->result_->type == kTfLiteFloat32) {
  //  return predictor->result_->data.f;
  //}else{
  //  return nullptr;
  //}

  return predictor->result_float_;
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
