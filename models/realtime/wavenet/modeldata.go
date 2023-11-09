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
	"github.com/nlpodyssey/waveny/floatreader"
	"os"
)

type ModelData struct {
	Version      string    `json:"version"`
	Architecture string    `json:"architecture"`
	Config       Config    `json:"config"`
	Weights      []float32 `json:"weights"`
}

func LoadFromJSONModelDataFile(filename string) (*Model, error) {
	modelData, err := ReadModelDataJSONFile(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to read JSON model data from file %q: %w", filename, err)
	}
	model, err := New(modelData.Config, floatreader.NewReader(modelData.Weights))
	if err != nil {
		return nil, fmt.Errorf("failed to initialize WaveNet from JSON configuration: %w", err)
	}
	return model, nil
}

func ReadModelDataJSONFile(filename string) (_ *ModelData, err error) {
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
	var modelData *ModelData
	if err = dec.Decode(&modelData); err != nil {
		return nil, fmt.Errorf("JSON decoding failed: %w", err)
	}

	// TODO: check version

	if modelData.Architecture != "WaveNet" {
		return nil, fmt.Errorf("only WaveNet architecture is supported, actual: %q", modelData.Architecture)
	}
	return modelData, nil
}
