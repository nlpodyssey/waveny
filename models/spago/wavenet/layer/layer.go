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

package layer

import (
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/spago/nn/activation"
	"github.com/nlpodyssey/waveny/models/spago/conv1d"
	"github.com/nlpodyssey/waveny/models/spago/conv1x1"
)

type Config struct {
	ConditionSize int
	Channels      int
	KernelSize    int
	Dilation      int
	Activation    string
	Gated         bool
}

type Model struct {
	nn.Module
	Conv       *conv1d.Model
	InputMixer *conv1d.Model
	Activation *activation.Model
	Conv1x1    *conv1x1.Model
	Gated      bool
}

func New(c Config) *Model {
	midChannels := c.Channels
	if c.Gated {
		midChannels *= 2
	}
	return &Model{
		Conv: conv1d.New(conv1d.Config{
			InChannels:  c.Channels,
			OutChannels: midChannels,
			KernelSize:  c.KernelSize,
			Dilation:    c.Dilation,
			Bias:        true,
		}),
		InputMixer: conv1d.New(conv1d.Config{
			InChannels:  c.ConditionSize,
			OutChannels: midChannels,
			KernelSize:  1,
			Dilation:    1,
			Bias:        false,
		}),
		Activation: activation.New(activation.MustParseActivation(c.Activation)),
		Conv1x1: conv1x1.New(conv1x1.Config{
			InChannels:  c.Channels,
			OutChannels: c.Channels,
			Bias:        true,
		}),
		Gated: c.Gated,
	}
}

func (m *Model) Forward(x, h mat.Tensor, outLength int) (toNextLayer, toMixer mat.Tensor) {
	zConv := m.Conv.Forward(x)
	im := m.InputMixer.Forward(h)
	zConvShape := zConv.Shape()
	imShape := im.Shape()
	z1 := ag.Add(zConv, ag.Slice(im, 0, imShape[1]-zConvShape[1], imShape[0], imShape[1]))
	postActivation := m.activation(z1)

	paShape := postActivation.Shape()
	xShape := x.Shape()
	xx := ag.Slice(x, 0, xShape[1]-paShape[1], xShape[0], xShape[1])
	aa := m.Conv1x1.Forward(postActivation)
	toNextLayer = ag.Add(xx, aa)
	toMixer = ag.Slice(postActivation, 0, paShape[1]-outLength, paShape[0], paShape[1])

	return toNextLayer, toMixer
}

func (m *Model) activation(z1 mat.Tensor) mat.Tensor {
	if !m.Gated {
		return m.Activation.Forward(z1)[0]
	}
	channels := m.Conv1x1.InChannels
	shape := z1.Shape()
	return ag.Mul(
		m.Activation.Forward(
			ag.Slice(z1, 0, 0, shape[0], channels),
		)[0],
		ag.Sigmoid(
			ag.Slice(z1, 0, channels, shape[0], shape[1]),
		),
	)
}

func (m *Model) ResetParameters() {
	m.Conv.ResetParameters()
	m.InputMixer.ResetParameters()
	m.Conv1x1.ResetParameters()
}
