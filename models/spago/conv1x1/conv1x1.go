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
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/waveny/models/spago/initializations"
	"math"
)

type Config struct {
	InChannels  int
	OutChannels int
	Bias        bool
}

type Model struct {
	nn.Module
	InChannels  int
	OutChannels int
	Weights     *nn.Param
	Bias        *nn.Param
}

func New(c Config) *Model {
	return &Model{
		InChannels:  c.InChannels,
		OutChannels: c.OutChannels,
		Weights:     nn.NewParam(mat.NewDense[float32](mat.WithShape(c.OutChannels, c.InChannels))),
		Bias:        makeBias(c),
	}
}

func makeBias(c Config) *nn.Param {
	if !c.Bias {
		return nil
	}
	return nn.NewParam(mat.NewDense[float32](mat.WithShape(c.OutChannels)))
}

func (m *Model) Forward(x mat.Tensor) mat.Tensor {
	y := ag.Mul(m.Weights, x)
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
	initializations.InitKaimingUniform(m.Weights, math.Sqrt(5), 1)
	if m.Bias == nil {
		return
	}
	if m.Bias != nil {
		fanIn := m.InChannels
		bound := 1 / math.Sqrt(float64(fanIn))
		initializations.InitUniform(m.Bias, -bound, bound)
	}
}
