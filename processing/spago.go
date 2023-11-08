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
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/waveny/models/spago/wavenet"
	"github.com/nlpodyssey/waveny/wave"
)

type SpagoConfig struct {
	ModelPath string
}

func ProcessWithSpagoModel(config Config, spagoConfig SpagoConfig) error {
	var model *wavenet.Model
	model, err := nn.LoadFromFile[*wavenet.Model](spagoConfig.ModelPath)
	if err != nil {
		return fmt.Errorf("failed to load SpaGO model %q: %w", spagoConfig.ModelPath, err)
	}

	input, err := wave.WavToSpagoTensor(config.InputPath)
	if err != nil {
		return err
	}

	output := model.Forward(input, true)
	return wave.SpagoTensorToWav(output, config.OutputPath)
}
