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
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/waveny/floats"
	"github.com/nlpodyssey/waveny/models/spago/initializations"
	"math"
)

type Config struct {
	InChannels  int
	OutChannels int
	KernelSize  int
	Dilation    int
	Bias        bool
}

type Model struct {
	nn.Module
	InChannels  int
	OutChannels int
	KernelSize  int
	Dilation    int
	Weights     []*nn.Param // [KernelSize](OutChannels x InChannels)
	Bias        *nn.Param   // Vector, optional
}

func New(s Config) *Model {
	return &Model{
		InChannels:  s.InChannels,
		OutChannels: s.OutChannels,
		KernelSize:  s.KernelSize,
		Dilation:    s.Dilation,
		Weights:     makeWeights(s),
		Bias:        makeBias(s),
	}
}

func makeWeights(s Config) []*nn.Param {
	weights := make([]*nn.Param, s.KernelSize)
	for i := range weights {
		weights[i] = nn.NewParam(mat.NewDense[float32](mat.WithShape(s.OutChannels, s.InChannels)))
	}
	return weights
}

func makeBias(s Config) *nn.Param {
	if !s.Bias {
		return nil
	}
	return nn.NewParam(mat.NewDense[float32](mat.WithShape(s.OutChannels)))
}

func (m *Model) Forward(x mat.Tensor) mat.Tensor {
	xShape := x.Shape()
	weights := m.Weights
	dilation := m.Dilation
	kernelSize := m.KernelSize

	outCols := xShape[1] - dilation*(kernelSize-1)

	y := ag.Mul(weights[0], ag.Slice(x, 0, 0, xShape[0], outCols))
	for i := 1; i < len(weights); i++ {
		offset := dilation * i
		y = ag.Add(y, ag.Mul(weights[i], ag.Slice(x, 0, offset, xShape[0], offset+outCols)))
	}

	if m.Bias != nil {
		// FIXME: wasting allocation
		cols := y.Shape()[1]
		extendedBias := ag.Mul(
			m.Bias,
			ag.StopGrad(mat.NewDense[float32](mat.WithShape(1, cols), mat.WithBacking(mat.CreateInitializedSlice[float32](cols, 1)))),
		)
		y = ag.Add(y, extendedBias)
	}
	return y
}

func (m *Model) ResetParameters() {
	for _, w := range m.Weights {
		initializations.InitKaimingUniform(w, math.Sqrt(5), m.KernelSize)
	}
	if m.Bias != nil {
		fanIn := m.InChannels * m.KernelSize
		bound := 1 / math.Sqrt(float64(fanIn))
		initializations.InitUniform(m.Bias, -bound, bound)
	}
}

func (m *Model) ExportParams(w *floats.Writer) {
	if len(m.Weights) > 0 {
		shape := m.Weights[0].Shape()
		outChannels := shape[0]
		inChannels := shape[1]

		for i := 0; i < outChannels; i++ {
			for j := 0; j < inChannels; j++ {
				for k := range m.Weights {
					w.Write(m.Weights[k].At(i, j).Item().F32())
				}
			}
		}
	}

	if m.Bias != nil {
		size := m.Bias.Size()
		for i := 0; i < size; i++ {
			w.Write(m.Bias.At(i).Item().F32())
		}
	}
}
