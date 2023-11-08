# Waveny

Waveny is a Go library and command-line utility designed for emulating guitar
amplifiers and pedals through deep learning.

The project takes inspiration from [Neural Amp Modeler] (NAM) and has
adapted significant components from related repositories into Go:

* [neural-amp-modeler]: a Python project for model training and WAVE file
  processing (reamping), leveraging PyTorch and Lightning.
* [NeuralAmpModelerCore]: the core DSP library, written in C++, suited to
  real-time plugin development.

## Development Status and Constraints

Waveny is in the early stages of development, and as such, the public APIs and
functionalities are subject to change.
The current codebase includes placeholders, indicating ongoing development,
and minimal documentation.

The project ambitiously applies digital signal processing in Go.
The `live` command aims to offer decent real-time processing on modern CPUs,
but optimization is ongoing.

Key technical constraints include:

* Sole support for the WaveNet model.
* Support limited to a 48kHz sample rate.
* Requirement for WAVE files to be PCM 48kHz/24-bit for training or reamping.

Future updates will address these limitations.

## Utilization Guide

### Command Line Interface

The `waveny` command-line interface offers several commands to interact with
the deep learning models for guitar amp emulation.

Build it with `go build ./cmd/waveny`, or compile-and-run it with
`go run ./cmd/waveny`.

The single executable allows to run different sub-commands, in this form:

```shell
waveny COMMAND [arguments...]
```

Run `waveny help` to see a list of the available commands. Here is a recap:

* `train`: train a new WaveNet model using SpaGO.
* `process-spago`: process a WAVE file using a pre-trained WaveNet SpaGO model,
  loaded from a file in "native" format.
* `process-rt`: process a WAVE file using the custom Waveny real-time-capable
  WaveNet model, loaded from a NAM model-data JSON file.
* `process-torch`: process a WAVE file using a WaveNet SpaGO model, loaded and
  converted from a pre-trained NAM PyTorch/Lightning checkpoint file.
* `live`: process audio input in real-time using the custom Waveny WaveNet
  model, loaded from a NAM model-data JSON file. It uses PortAudio for I/O.

For detailed usage and arguments of each command, execute:

```shell
waveny COMMAND -h
```

### Library Integration

Integrate Waveny as a Go module with:

```shell
go get github.com/nlpodyssey/waveny
```

From now on, we will refer to the root package `github.com/nlpodyssey/waveny`
as simply `waveny`.

Currently, the library only implements the WaveNet deep learning network.
Under `waveny/models` you will find two different model implementations.

Package `waveny/models/spago/wavenet` implements the model with [SpaGO] machine
learning library.

A WaveNet SpaGO model can be trained from scratch -
see `waveny/models/spago/wavenet/training` subpackage.

It's also possible to load a PyTorch model file, pre-trained with
[neural-amp-modeler] Python project, and convert it into a SpaGO model -
see `waveny/models/spago/wavenet/torchconv` subpackage.
It uses [GoPickle] library to read torch models without the need to run Python.

A pre-trained SpaGO model can be effectively used to process WAVE files
(non-real-time reamping). It is less suitable for real-time processing,
mostly due to memory allocations and usage of goroutines.

For real-time use, we provide another custom implementation
of WaveNet, in package `waveny/models/realtime/wavenet`.

The real-time-capable model can load NAM plugin model files, in JSON format,
such as WaveNet models from [ToneHunt].

This implementation takes advantage of a self-contained package for handling
matrices and vectors, implemented in `waveny/models/realtime/mat`.
Inspired by the original [NeuralAmpModelerCore] implementation, and the
underlying [Eigen] library, it allows to minimize the amount of memory
allocations, permitting a predictable execution time, suitable for real-time
processing.

Package `waveny/liveplay` implements real-time processing procedures,
using [PortAudio] go bindings for I/O.

Package `waveny/wave` provides utilities for reading and writing WAVE files.

[SpaGO]: https://github.com/nlpodyssey/spago
[GoPickle]: https://github.com/nlpodyssey/gopickle
[ToneHunt]: https://tonehunt.org
[Neural Amp Modeler]: https://www.neuralampmodeler.com
[neural-amp-modeler]: https://github.com/sdatkinson/neural-amp-modeler
[NeuralAmpModelerCore]: https://github.com/sdatkinson/NeuralAmpModelerCore
[Eigen]: https://eigen.tuxfamily.org
[PortAudio]: https://github.com/gordonklaus/portaudio
