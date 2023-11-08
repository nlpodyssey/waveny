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

package processing

import (
	"github.com/nlpodyssey/waveny/models/realtime/wavenet"
	"github.com/nlpodyssey/waveny/wave"
)

type RTConfig struct {
	ModelDataPath string
}

func ProcessWithRTModel(config Config, rtConfig RTConfig) error {
	model, err := wavenet.LoadFromJSONModelDataFile(rtConfig.ModelDataPath)
	if err != nil {
		return err
	}

	input, err := wave.WavToFloats(config.InputPath)
	if err != nil {
		return err
	}

	output := make([]float32, len(input))

	const chunkSize = 4096
	numChunks := len(input) / chunkSize

	for i := 0; i < numChunks; i++ {
		from := i * chunkSize
		to := from + chunkSize

		model.Process(input[from:to], output[from:to])
		model.Finalize(chunkSize)
	}

	if from := chunkSize * numChunks; from < len(input) {
		model.Process(input[from:], output[from:])
		model.Finalize(len(input) - from)
	}

	return wave.FloatsToWav(output, config.OutputPath)
}
