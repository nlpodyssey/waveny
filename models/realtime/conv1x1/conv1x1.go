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

package conv1x1

import (
	"github.com/nlpodyssey/waveny/models/realtime/floatreader"
	"github.com/nlpodyssey/waveny/models/realtime/mat"
)

type Config struct {
	InChannels  int
	OutChannels int
	Bias        bool
}

type Model struct {
	weight  mat.Matrix
	bias    mat.Vector
	hasBias bool
}

func New(config Config) *Model {
	c := &Model{
		weight:  mat.NewMatrix(config.OutChannels, config.InChannels),
		hasBias: config.Bias,
	}
	if config.Bias {
		c.bias = mat.NewVector(config.OutChannels)
	}
	return c
}

func (m *Model) GetOutChannels() int {
	return m.weight.Rows()
}

func (m *Model) SetParams(params *floatreader.Reader) {
	for i := 0; i < m.weight.Rows(); i++ {
		for j := 0; j < m.weight.Columns(); j++ {
			m.weight.Set(i, j, params.Next())
		}
	}
	if m.hasBias {
		for i := 0; i < m.bias.Size(); i++ {
			m.bias.Set(i, params.Next())
		}
	}
}

func (m *Model) Process(input, output mat.Matrix) {
	mat.Product(m.weight, input, output)
	if m.hasBias {
		mat.AddInPlaceColumnWise(output, m.bias)
	}
}