package tflite

// #cgo CXXFLAGS: -std=c++11 -I${SRCDIR}/cbits -O3 -Wall -g -Wno-sign-compare -Wno-unused-function  -I/home/as29/my_tflite/tensorflow/bazel-tensorflow/external/flatbuffers/include -I/home/as29/my_tflite/tensorflow/bazel-tensorflow/external/com_google_absl -I/home/as29/my_tflite/tensorflow
// #cgo LDFLAGS: -lstdc++ -L/home/as29/my_android_ndk/android-ndk-r19c -llog -L/opt/tflite/lib -ltensorflowlite -lallocation -larena_planner -lexternal_cpu_backend_context -lframework -lgl_delegate -lminimal_logging -lnnapi_delegate -lsimple_memory_arena -lstring_util -ltime -lutil
import "C"
