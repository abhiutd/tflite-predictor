#define _GLIBCXX_USE_CXX11_ABI 0

#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <iostream>

#include "tensorflow/contrib/lite/interpreter.h"
#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/model.h"
#include "tensorflow/contrib/lite/optional_debug_tools.h"

// ISSUE: would make it tensorflow specific 
//#include "tensorflow/contrib/android/asset_manager_filesystem.cc"

#include "predictor.hpp"

#if 0
#define DEBUG_STMT std ::cout << __func__ << "  " << __LINE__ << "\n";
#else
#define DEBUG_STMT
#endif

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
};

Predictor::Predictor(const string &model_file, int batch, int mode) {
	
  /* Load the network. */
	// Tflite uses FlatBufferModel format to store/access model instead of protobuf unlike tensorflow
	char* model_file_char = const_cast<char*>(model_file.c_str());
	net_ = tflite::FlatBufferModel::BuildFromFile(model_file_char);
	
  // Build interpreter
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder(*net_, resolver)(&interpreter);
	
	assert(net_ != nullptr);
	assert(interpreter != nullptr);

	mode_ = mode;
	batch_ = batch;
}

void Predictor::Predict(float* inputData) {
	
  // Branch out based on chosen hardware mode
  if(mode_ == 0) {
    // CPU (default)
	  interpreter->SetNumThreads(4);
  }else if(mode_ == 1) {
    // GPU
    auto* delegate =TfLiteGpuDelegateCreate(/*options=*/nullptr);
    if(interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk) return;
  }else if(mode_ == 2) {
    // NNAPI (decides internally which hardware mode to run inference in)
    interpreter->setUseNNAPI(true);
  }

	// allocate tensor buffers
	if(!(interpreter->AllocateTensors() == kTfLiteOk)) {
		fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__);
		exit(1);
	}

	// fill input buffers
	const int input = interpreter->inputs()[0];
	TfLiteTensor* input_tensor = interpreter->tensor(input);
	TfLiteIntArray* input_dims = input_tensor->dims;
	width_ = input_dims->data[1];
	height_ = input_dims->data[2];
	channels_ = input_dims->data[3];

	assert(input_dims->size == 4);
	const int size = batch_ * width_ * height_ * channels_;
	memcpy(input_tensor->data.f, &inputData[0], size);

	const int output = interpreter->outputs()[0];
	result_ = interpreter->tensor(output);

	// run inference
	if(!(interpreter->Invoke() == kTfLiteOk)) {
		fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__);
		exit(1);
	}

	// Note: TfLiteTensor does not provide a size() API call which means we have to fetch the number of bytes the tensor
	// has and divide it by 4 since we assume float is 4 bytes long
	// Potential Bug location
	//pred_len_ = result_->bytes/(4*batch_);
	assert(result_->dims->size == 2);
	assert(result_->dims->data[0] == 1);
	pred_len_ = result_->dims->data[1];
	// DEBUG for result_
	if(!(result_->type == kTfLiteFloat32)) {
		fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__);
    exit(1);
	}
	
  if(result_->data.f == nullptr) {
		throw std::runtime_error("expected a non-nil result in Predict()");
	}
}

PredictorContext NewTflite(char *model_file, int batch, int mode) {
	try {
		
    const auto ctx = new Predictor(model_file, batch, mode);
		return (void *)ctx;
	}catch (const std::invalid_argument &ex) {
		//LOG(ERROR) << "exception: " << ex.what();
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
