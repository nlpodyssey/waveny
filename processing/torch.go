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
	"fmt"
	"github.com/nlpodyssey/waveny/models/spago/wavenet"
	"github.com/nlpodyssey/waveny/models/spago/wavenet/torchconv"
	"github.com/nlpodyssey/waveny/wave"
)

type TorchConfig struct {
	ConfigPath string
	ModelPath  string
}

func ProcessWithTorchModel(config Config, torchConfig TorchConfig) error {
	modelConfig, err := wavenet.ReadModelConfigJSONFile(torchConfig.ConfigPath)
	if err != nil {
		return fmt.Errorf("failed to read JSON model config from file %q: %w", torchConfig.ConfigPath, err)
	}

	model := wavenet.New(modelConfig.Net.Config)
	err = torchconv.LoadTorchModel(torchConfig.ModelPath, model)
	if err != nil {
		return fmt.Errorf("failed to load-and-convert torch model from file %q: %w", torchConfig.ModelPath, err)
	}

	input, err := wave.WavToSpagoTensor(config.InputPath)
	if err != nil {
		return err
	}

	output := model.Forward(input, true)
	return wave.SpagoTensorToWav(output, config.OutputPath)
}
