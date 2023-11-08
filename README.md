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
* Requirement for WAVE files to be PCM 48kHz 24-bit mono, for training or
  reamping.
* Training on CPU only.

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

Run `waveny help` to see a list of available commands. Here is a recap:

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

The following sections illustrate several use cases.

#### Training

To train a new WaveNet SpaGO model from scratch, we first need to prepare
some files.

* A clean/unprocessed audio file (WAVE PCM 48kHz 24-bit mono). You are free to
  create or record your own. However, [neural-amp-modeler] Python project
  comes with a set of standardized files: see [this section](https://github.com/sdatkinson/neural-amp-modeler/blob/v0.7.3/README.md#standardized-reamping-files)
  from the README. We recomment to use v3.0.0.
* A "reamped" target audio file (same format as above), that is, the audio
  signal from the previous file processed through the real amp or pedal that
  you want to emulate.
* A JSON configuration that describes structure and shapes of the WaveNet model.
  A good starting point is this [wavenet.json](https://github.com/sdatkinson/neural-amp-modeler/blob/v0.7.3/bin/train/inputs/models/wavenet.json)
  file from NAM Python project.

Clean and reamped files must have the same size and be perfectly aligned.
Their coupled audio content will be split into a training set and a validation
set: the splitting points can be specified with dedicated command line
arguments. If you are using NAM standardized reamping file
[v3_0_0.wav](https://drive.google.com/file/d/1Pgf8PdE0rKB1TD4TRPKbpNo1ByR3IOm9/view?usp=drive_link),
then the default values are already a good fit.

Please have a look at the output of `waveny train -h` to learn about additional
arguments.

Here is a minimal example:

```shell
waveny train \
  -config path/to/wavenet.json \
  -input path/to/v3_0_0.wav \
  -target path/to/v3_0_0_reamped.wav \
  -out path/to/output-folder/
```

A new sub-folder, named with the current time-stamp, is created under the
specified output path. Here, at the end of each training epoch, a checkpoint
SpaGO model file is created. Loss values are shown for each epoch, and
they also become part of the checkpoint file names, for convenience. 

You can stop the training manually at any moment with Ctrl+C.

When you are satisfied with the accuracy, pick the best model file from
the output folder, and see further steps described below.

#### Process an audio file with pre-trained models

##### Loading a SpaGO model

You can use a pre-trained Waveny SpaGO model to process an audio file.

Get or record a "clean" audio file (WAVE PCM 48kHz 24-bit mono), choose
your preferred model, and run:

```shell
waveny process-spago \
  -input path/to/input.wav \
  -output path/to/output.wav \
  -model path/to/waveny.model
```

This command creates the output WAVE file, processing your input with the
given amp/pedal emulation model.

##### Loading a PyTorch model, running with SpaGO

If you trained a WaveNet model with [neural-amp-modeler] Python project,
you can use another command to load a PyTorch/Lightning model/checkpoint file,
convert it into a corresponding SpaGO model (on-the-fly, in memory), and
process an input file just like we did above. In this case, you also need the
accompanying WaveNet JSON model-data description. For example:

```shell
waveny process-torch \
  -input path/to/input.wav \
  -output path/to/output.wav \
  -config path/to/config_model.json \
  -model path/torch_model.ckpt
```

##### Loading a NAM model, running with Waveny custom implementation

There is yet another way to process an audio file. This time, we are going to
provide a model in NAM "exported" JSON format. This is the most common format
used to share NAM models, intended to work with existing NAM audio plugins.

An amazing source for pre-trained models in this special format is [ToneHunt]
website. Download your desired amp/pedal emulation model, and make sure
it uses WaveNet architecture. Then run:

```shell
waveny process-rt \
  -input path/to/input.wav \
  -output path/to/output.wav \
  -model path/to/model.nam
```

The "rt" suffix in the command name indicates that we are using a custom
WaveNet DSP processor. This implementation is most suitable for real-time
processing, a topic discussed in the next section.

#### Process audio in real-time

The same kind of NAM models involved in our last example can also be used for
real-time processing. Plug in your musical instrument, and run:

```shell
waveny live -model path/to/model.nam
```

This command uses Waveny custom WaveNet implementation to process audio input
in real-time. I/O is possible thanks to [PortAudio].

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
