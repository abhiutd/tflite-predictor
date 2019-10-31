#ifndef __PREDICTOR_HPP__
#define __PREDICTOR_HPP__

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#include <stddef.h>
#include <stdbool.h>

typedef void *PredictorContext;

PredictorContext NewTflite(char *model_file, int batch, int mode, bool verbose, bool profile);

void SetModeTflite(int mode);

void InitTflite();

void PredictTflite(PredictorContext pred, int* inputData_quantize, float* inputData_float, bool quantize);

float* GetPredictionsTflite(PredictorContext pred);

void DeleteTflite(PredictorContext pred);

int GetWidthTflite(PredictorContext pred);

int GetHeightTflite(PredictorContext pred);

int GetChannelsTflite(PredictorContext pred);

int GetPredLenTflite(PredictorContext pred);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // __PREDICTOR_HPP__
