package tflite

// #cgo CXXFLAGS: -std=c++11 -I${SRCDIR}/cbits -O3 -Wall -g -Wno-sign-compare -Wno-unused-function  -I/Users/abhiutd/workspace/scratch/tflite/tensorflow -I/Users/abhiutd/workspace/scratch/flatbuffers/flatbuffers/include
// #cgo LDFLAGS: -lstdc++ -L/Users/abhiutd/workspace/scratch/android-ndk -llog -L/Users/abhiutd/workspace/scratch/tflite/lib/tensorflow/contrib/lite -ltflite -lframework -L/Users/abhiutd/workspace/scratch/flatbuffers/flatbuffers
import "C"
