// Copyright 2023 The NLP Odyssey Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package cli

const usage = `Usage:

  waveny COMMAND [arguments...]

List of commands:

  help, -help, --help, -h
    Print usage information and exit.

  train
    Train a new WaveNet model using SpaGO, producing both SpaGO and .nam models.

  process-spago
    Process a WAVE file using a pre-trained WaveNet SpaGO model,
    loaded from a file in "native" format.

  process-rt
    Process a WAVE file using the custom Waveny real-time-capable
    WaveNet model, loaded from a .nam model-data file.

  process-torch
    Process a WAVE file using a WaveNet SpaGO model, loaded and converted
    from a pre-trained NAM PyTorch/Lightning checkpoint file.

  live
    Process audio input in real-time using the custom Waveny WaveNet
    model, loaded from a .nam model-data file. It uses PortAudio for I/O.

For detailed usage and arguments of each command, execute:

  waveny COMMAND -h
`
