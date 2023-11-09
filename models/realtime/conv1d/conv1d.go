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

package conv1d

import (
	"github.com/nlpodyssey/waveny/floatreader"
	"github.com/nlpodyssey/waveny/models/realtime/mat"
)

type Config struct {
	InChannels  int
	OutChannels int
	KernelSize  int
	Bias        bool
	Dilation    int
}

type Model struct {
	weight   []mat.Matrix // [kernel](OutChannels, InChannels)
	bias     mat.Vector
	dilation int
	hasBias  bool
}

func New(config Config) *Model {
	return &Model{
		weight:   makeWeight(config),
		bias:     makeBias(config),
		dilation: config.Dilation,
		hasBias:  config.Bias,
	}
}

func makeWeight(config Config) []mat.Matrix {
	weight := make([]mat.Matrix, config.KernelSize)
	for i := range weight {
		weight[i] = mat.NewMatrix(config.OutChannels, config.InChannels)
	}
	return weight
}

func makeBias(config Config) mat.Vector {
	if !config.Bias {
		return mat.Vector{}
	}
	return mat.NewVector(config.OutChannels)
}

func (m *Model) SetParams(params *floatreader.Reader) {
	if len(m.weight) > 0 {
		outChannels := m.weight[0].Rows()
		inChannels := m.weight[0].Columns()

		for i := 0; i < outChannels; i++ {
			for j := 0; j < inChannels; j++ {
				for k := range m.weight {
					m.weight[k].Set(i, j, params.Next())
				}
			}
		}
	}

	if m.hasBias {
		for i := 0; i < m.bias.Size(); i++ {
			m.bias.Set(i, params.Next())
		}
	}
}

func (m *Model) GetInChannels() int {
	if len(m.weight) == 0 {
		return 0
	}
	return m.weight[0].Columns()
}

func (m *Model) GetOutChannels() int {
	if len(m.weight) == 0 {
		return 0
	}
	return m.weight[0].Rows()
}

func (m *Model) GetKernelSize() int {
	return len(m.weight)
}

func (m *Model) GetDilation() int {
	return m.dilation
}

func (m *Model) GetNumParams() int {
	return m.dilation
}

func (m *Model) Process(input, output mat.Matrix, inputStartColumn, numColumns, outputStartColumn int) {
	weight := m.weight
	kernelSize := len(weight)
	dilation := m.dilation

	offset := dilation * (1 - kernelSize)
	mat.Product(
		weight[0],
		input.ViewMiddleColumns(inputStartColumn+offset, numColumns),
		output.ViewMiddleColumns(outputStartColumn, numColumns),
	)

	for k := 1; k < kernelSize; k++ {
		offset = dilation * (k + 1 - kernelSize)
		mat.AddProduct(
			weight[k],
			input.ViewMiddleColumns(inputStartColumn+offset, numColumns),
			output.ViewMiddleColumns(outputStartColumn, numColumns),
		)
	}

	if m.hasBias {
		mat.AddInPlaceColumnWise(
			output.ViewMiddleColumns(outputStartColumn, numColumns),
			m.bias,
		)
	}
}
