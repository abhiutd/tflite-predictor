# go-tflite

[![Go Report Card](https://goreportcard.com/badge/github.com/rai-project/go-mxnet)](https://goreportcard.com/report/github.com/rai-project/go-mxnet)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Go binding for Tensorflow Lite C++ API. It is also referred to as MLModelScope Tensorflow Lite mobile Predictor (TFLite mPredictor). It is used to perform model inference on mobile devices. It is used by the [Tensorflow Lite agent](https://github.com/abhiutd/tflite-agent) in [MLModelScope](mlmodelscope.org) to perform model inference in Go. More importantly, it can be used as a standalone predictor in any given Android/iOS application. Refer to [Usage Modes](Usage Modes) for further details.

## Installation

Download and install go-mxnet:

```
go get -v github.com/abhiutd/tflite-predictor
```

The binding requires Tensorflow Lite, Gomobile and other Go packages.

### Tensorflow Lite C++ Library

The Tensorflow Lite C++ library is expected to be under `/opt/tflite`.

Note that the mPredictor requires shared library of Tensorflow Lite rather than Java/Objective-C/Swift compatible builds which is the conventional way. This implies one needs to build it from source. Kindly follow aforementioned steps to do so as Tensorflow Lite does not provide a formal documentation for it. 

1. Install Bazel

Install appropriate version of Bazel as per Tensorflow documentation [Tensorflow Install](https://www.tensorflow.org/install/source).

2. Download Tensorflow Lite source code

```
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
```

3. Checkout appropriate release branch

```
git checkout branch_name  # r1.9, r1.10, etc.
```

4. Add shared library build target

```
cd tensorflow/lite
vim BUILD
```

Add the following to the end of `BUILD` file, if not already present.

```
tflite_cc_shared_object(
    name = "libtensorflowlite.so",
    linkopts = select({
        "//tensorflow:macos": [
            "-Wl,-exported_symbols_list,$(location //tensorflow/lite:tflite_exported_symbols.lds)",
            "-Wl,-install_name,@rpath/libtensorflowlite.so",
        ],
        "//tensorflow:windows": [],
        "//conditions:default": [
            "-z defs",
            "-Wl,--version-script,$(location //tensorflow/lite:tflite_version_script.lds)",
        ],
    }),
    deps = [
        ":framework",
        ":tflite_exported_symbols.lds",
        ":tflite_version_script.lds",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/delegates/gpu:gl_delegate",
        "//tensorflow/lite/delegates/nnapi:nnapi_delegate",
        "//tensorflow/lite/profiling:profiler",
        "//tensorflow/lite/tools/evaluation:utils",
        "@com_google_absl//absl/memory",
    ],
)
``` 

```
cd ../../
```

5. Configure the build

```
./configure
```

Configure Tensorflow Lite build as guided by the script. Make sure to provide appropriate version and library path of Android NDK and SDK, and mark `yes/y` for Android build if building for Android. For faster build, mark `No/n` for rest of the dependencies. 

6. Build package

```
bazel build -c opt //tensorflow/lite:libtensorflowlite.so --crosstool_top=//external:android/crosstool --host_crosstool_top=@bazel_tools//tools/cpp:toolchain --config=android_arm64 --cpu=arm64-v8a --fat_apk_cpu=arm64-v8a
```

Given `--cpu` and `--fat_apk_cpu` options build for `arm64-v8a` ISA. Alter the options as per requirement. Copy required header and library files to `/opt/tflite`. See [lib.go](lib.go) for details. For instance, Tensorflow Lite mPredictor also depends on `Google Flatbuffer` (found as part of Tensorflow Lite repository), `libEGL.so` and `libGLESv3.so` (found as part of Android NDK) and so on. 

If you get an error about not being able to write to `/opt` then perform the following

```
sudo mkdir -p /opt/tflite
sudo chown -R `whoami` /opt/tflite
```

If you are using custom path for build files, change CGO_CFLAGS, CGO_CXXFLAGS and CGO_LDFLAGS enviroment variables. Refer to [Using cgo with the go command](https://golang.org/cmd/cgo/#hdr-Using_cgo_with_the_go_command).

For example,

```
    export CGO_CFLAGS="${CGO_CFLAGS} -I/tmp/tflite/include"
    export CGO_CXXFLAGS="${CGO_CXXFLAGS} -I/tmp/tflite/include"
    export CGO_LDFLAGS="${CGO_LDFLAGS} -L/tmp/tflite/lib"
```

### Go Packages

You can install the dependency through `go get`.

```
cd $GOPATH/src/github.com/abhiutd/tflite-predictor
go get -u -v ./...
```

Or use [Dep](https://github.com/golang/dep).

```
dep ensure -v
```

This installs the dependency in `vendor/`. It is the preferred option.

Also, one needs to install `gomobile` to be able to generate Java/Objective-C bindings of the mPredictor. 

```
go get golang.org/x/mobile/cmd/gomobile
gomobile init
```

### Configure Environmental Variables

Configure the linker environmental variables since the Tensorflow Lite C++ library is under a non-system directory. Place the following in either your `~/.bashrc` or `~/.zshrc` file

Linux
```
export LIBRARY_PATH=$LIBRARY_PATH:/opt/tflite/lib
export LD_LIBRARY_PATH=/opt/tflite/lib:$DYLD_LIBRARY_PATH

```

macOS
```
export LIBRARY_PATH=$LIBRARY_PATH:/opt/tflite/lib
export DYLD_LIBRARY_PATH=/opt/tflite/lib:$DYLD_LIBRARY_PATH
```

### Generate bindings

Tensorflow Lite mPredictor is written in Go, binded with Tensorflow Lite C++ API. To be able to use it in a mobile application, you would have to generate appropriate bindings (Java for Android and Objective-C for iOS). We provide bindings off-the-shelf in [bindings](bindings), but you can generate your own by using the following command.

```
gomobile bind -o bindings/android/tflite-predictor.aar -target=android/arm64 -v github.com/abhiutd/tflite-predictor
```

This command builds `tflite-predictor.aar` binary for Android with `arm64` ISA. Change it to `tflite-predictor.framework` and approrpiate ISA for iOS.

### Usage Modes

One can employ Tensorflow Lite mPredictor to perform model inference in multiple ways, which are listed below.

1. Standalone Predictor (mPredictor)

There are four main API calls to be used for performing model inference in a given mobile application.

```
// create a Tensorflow Lite mPredictor
New()

// perform inference on given input data
Predict()

// generate output predictions
ReadPredictedOutputFeatures()

// delete the Tensorflow Lite mPredictor
Close()
```

Refer to [cbits.go](cbits.go) for details on the inputs/outputs of each API call.

2.  MLModelScope Mobile Agent

Download MLModelScope mobile agent from [agent](https://github.com/abhiutd/agent-classification-android). It has Tensorflow Lite and Qualcomm SNPE mPredictors in built. Refer to its documentation to understand its usage.

3. MLModelScope web UI

Choose Tensorflow Lite as framework and one of the available mobile devices as hardware backend to perform model inference through web interface.
