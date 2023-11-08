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

package wavenet

import (
	"encoding/json"
	"fmt"
	"os"
)

// A ModelConfig specifies the configuration for a model, and other settings
// for training. The fields are matching the JSON model configuration.
// Only WaveNet model is supported.
type ModelConfig struct {
	Net       NetConfig       `json:"net"`
	Optimizer OptimizerConfig `json:"optimizer"`
	Scheduler SchedulerConfig `json:"lr_scheduler"`
}

type NetConfig struct {
	Name   string `json:"name"`
	Config Config `json:"config"`
}

type OptimizerConfig struct {
	LR float32 `json:"lr"`
}

type SchedulerConfig struct {
	Class string              `json:"class"`
	Args  SchedulerArgsConfig `json:"kwargs"`
}

type SchedulerArgsConfig struct {
	Gamma float32 `json:"gamma"`
}

func ReadModelConfigJSONFile(filename string) (_ *ModelConfig, err error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}
	defer func() {
		if e := file.Close(); e != nil && err == nil {
			err = fmt.Errorf("failed to close file: %w", e)
		}
	}()

	dec := json.NewDecoder(file)
	var modelConfig *ModelConfig
	if err = dec.Decode(&modelConfig); err != nil {
		return nil, fmt.Errorf("JSON decoding failed: %w", err)
	}
	if modelConfig.Net.Name != "WaveNet" {
		return nil, fmt.Errorf("only WaveNet is supported, actual: %q", modelConfig.Net.Name)
	}
	return modelConfig, nil
}
