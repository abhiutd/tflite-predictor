package tflite

// #cgo CXXFLAGS: -std=c++11 -I${SRCDIR}/cbits -O3 -Wall -g -Wno-sign-compare -Wno-unused-function  -I/Users/abhiutd/workspace/mobile/tensorflow
// #cgo LDFLAGS: -lstdc++ -L/Users/abhiutd/workspace/mobile/android-ndk-r18b -llog -L//Users/abhiutd/workspace/mobile/lib -ltensorflowlite -lframework -lnnapi_delegate
import "C"
