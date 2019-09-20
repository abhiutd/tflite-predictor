package tflite

// #include <stdio.h>
// #include <stdlib.h>
// #include "cbits/predictor.hpp"
import "C"
import (
	"fmt"
	"unsafe"
	"bufio"
	"os"
	"sort"

	"github.com/Unknwon/com"
	"github.com/pkg/errors"
	"github.com/rai-project/dlframework"
	"github.com/rai-project/dlframework/framework/feature"
)

// Hardware Modes
const (
	CPUMode        = 0
	GPUMode        = 1
	NNAPIMode      = 2
)

// Predictor Structure definition
type PredictorData struct {
	ctx   C.PredictorContext
	mode  int
	batch int
}

// Make access to mode and batch public
func (pd *PredictorData) Inc() {
	pd.mode++
	pd.batch++
}

// Create new Predictor Structure
func NewPredictorData() *PredictorData {
	return &PredictorData{}
}

// Create new predictor
func New(model string, mode, batch int) (*PredictorData, error) {

	modelFile := model
	if !com.IsFile(modelFile) {
		return nil, errors.Errorf("file %s not found", modelFile)
	}

	return &PredictorData{
		ctx: C.NewTflite(
			C.CString(modelFile),
			C.int(batch),
			C.int(mode),
		),
		mode:  mode,
		batch: batch,
	}, nil
}

// Initialize TFLite
func init() {
	C.InitTflite()
}

// Run inference
func Predict(p *PredictorData, data []byte) error {

	if len(data) == 0 {
		return fmt.Errorf("image data is empty")
	}

	ptr := (*C.float)(unsafe.Pointer(&data[0]))

	C.PredictTflite(p.ctx, ptr)

	return nil
}

// Return Top-1 predicted label
func ReadPredictionOutput(p *PredictorData, labelFile string) (string, error) {

	batchSize := p.batch
	if batchSize == 0 {
		return "", errors.New("null batch")
	}

	predLen := int(C.GetPredLenTflite(p.ctx))
	if predLen == 0 {
		return "", errors.New("null predLen")
	}
	
  length := batchSize * predLen
	if p.ctx == nil {
		return "", errors.New("empty predictor context")
	}
	
  cPredictions := C.GetPredictionsTflite(p.ctx)

	if cPredictions == nil {
		return "", errors.New("empty predictions")
	}

	slice := (*[1 << 15]float32)(unsafe.Pointer(cPredictions))[:length:length]

	var labels []string
	f, err := os.Open(labelFile)
	if err != nil {
		panic(err)
	}

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		labels = append(labels, line)
	}

	features := make([]dlframework.Features, batchSize)
	featuresLen := len(slice) / batchSize

	for ii := 0; ii < batchSize; ii++ {
		rprobs := make([]*dlframework.Feature, featuresLen)
		for jj := 0; jj < featuresLen; jj++ {
			rprobs[jj] = feature.New(
				feature.ClassificationIndex(int32(jj)),
				feature.ClassificationLabel(labels[jj]),
				feature.Probability(slice[ii*featuresLen+jj]),
			)
		}
		sort.Sort(dlframework.Features(rprobs))
		features[ii] = rprobs
	}

	top1 := features[0][0]
	return top1.GetClassification().GetLabel(), nil

}

// Delete the predictor 
func Close(p *PredictorData) {
	C.DeleteTflite(p.ctx)
}
