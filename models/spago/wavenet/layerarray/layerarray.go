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

package layerarray

import (
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/waveny/floats"
	rtlayerarray "github.com/nlpodyssey/waveny/models/realtime/wavenet/layerarray"
	"github.com/nlpodyssey/waveny/models/spago/conv1d"
	"github.com/nlpodyssey/waveny/models/spago/wavenet/layer"
)

// A Config specifies the configuration for instantiating a new layer-array Model.
// The fields are matching the JSON model configuration.
type Config struct {
	ConditionSize int    `json:"condition_size"`
	InputSize     int    `json:"input_size"`
	Channels      int    `json:"channels"`
	HeadSize      int    `json:"head_size"`
	KernelSize    int    `json:"kernel_size"`
	Dilations     []int  `json:"dilations"`
	Activation    string `json:"activation"`
	Gated         bool   `json:"gated"`
	HeadBias      bool   `json:"head_bias"`
}

type Model struct {
	nn.Module
	Config         Config
	Rechannel      *conv1d.Model
	Layers         []*layer.Model
	HeadRechannel  *conv1d.Model
	ReceptiveField int
}

func New(c Config) *Model {
	return &Model{
		Config: c,
		Rechannel: conv1d.New(conv1d.Config{
			InChannels:  c.InputSize,
			OutChannels: c.Channels,
			KernelSize:  1,
			Dilation:    1,
			Bias:        false,
		}),
		Layers: makeLayers(c),
		HeadRechannel: conv1d.New(conv1d.Config{
			InChannels:  c.Channels,
			OutChannels: c.HeadSize,
			KernelSize:  1,
			Dilation:    1,
			Bias:        c.HeadBias,
		}),
		ReceptiveField: 1 + (c.KernelSize-1)*sumInts(c.Dilations),
	}
}

func sumInts(values []int) int {
	s := 0
	for _, d := range values {
		s += d
	}
	return s
}

func makeLayers(c Config) []*layer.Model {
	layers := make([]*layer.Model, len(c.Dilations))
	for i, dilation := range c.Dilations {
		layers[i] = layer.New(layer.Config{
			ConditionSize: c.ConditionSize,
			Channels:      c.Channels,
			KernelSize:    c.KernelSize,
			Dilation:      dilation,
			Activation:    c.Activation,
			Gated:         c.Gated,
		})
	}
	return layers
}

func (m *Model) Forward(x, c, headInput mat.Tensor) (headInputOut, y mat.Tensor) {
	outLength := x.Shape()[1] - (m.ReceptiveField - 1)
	x = m.Rechannel.Forward(x)
	for _, l := range m.Layers {
		var headTerm mat.Tensor
		x, headTerm = l.Forward(x, c, outLength)
		if headInput == nil {
			headInput = headTerm
		} else {
			shape := headInput.Shape()
			headInput = ag.Add(
				ag.Slice(headInput, 0, shape[1]-outLength, shape[0], shape[1]),
				headTerm,
			)
		}
	}
	return m.HeadRechannel.Forward(headInput), x
}

func (m *Model) ResetParameters() {
	m.Rechannel.ResetParameters()
	m.HeadRechannel.ResetParameters()
	for _, l := range m.Layers {
		l.ResetParameters()
	}
}

func (m *Model) ExportConfig() rtlayerarray.Config {
	return rtlayerarray.Config{
		InputSize:     m.Config.InputSize,
		ConditionSize: m.Config.ConditionSize,
		HeadSize:      m.Config.HeadSize,
		Channels:      m.Config.Channels,
		KernelSize:    m.Config.KernelSize,
		Dilations:     m.Config.Dilations,
		Activation:    m.Config.Activation,
		Gated:         m.Config.Gated,
		HeadBias:      m.Config.HeadBias,
	}
}

func (m *Model) ExportParams(w *floats.Writer) {
	m.Rechannel.ExportParams(w)
	for _, l := range m.Layers {
		l.ExportParams(w)
	}
	m.HeadRechannel.ExportParams(w)
}
