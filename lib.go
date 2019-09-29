package tflite

// #cgo CXXFLAGS: -std=c++11 -I${SRCDIR}/cbits -O3 -Wall -g -Wno-sign-compare -Wno-unused-function  -I/Users/abhiutd/workspace/mobile/tensorflow/bazel-tensorflow/external/flatbuffers/include -I/Users/abhiutd/workspace/mobile/tensorflow
// #cgo LDFLAGS: -lstdc++ -L/Users/abhiutd/workspace/mobile/android-ndk-r18b -llog -L/Users/abhiutd/workspace/mobile/lib -ltensorflowlite -lgl_delegate -lminimal_logging -lnnapi_delegate -lnnapi_implementation -ltime -lutil
import "C"
