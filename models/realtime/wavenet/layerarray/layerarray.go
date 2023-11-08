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
	"github.com/nlpodyssey/waveny/models/realtime/conv1x1"
	"github.com/nlpodyssey/waveny/models/realtime/floatreader"
	"github.com/nlpodyssey/waveny/models/realtime/mat"
	"github.com/nlpodyssey/waveny/models/realtime/wavenet/layer"
)

type Config struct {
	InputSize     int    `json:"input_size"`
	ConditionSize int    `json:"condition_size"`
	HeadSize      int    `json:"head_size"`
	Channels      int    `json:"channels"`
	KernelSize    int    `json:"kernel_size"`
	Dilations     []int  `json:"dilations"`
	Activation    string `json:"activation"`
	Gated         bool   `json:"gated"`
	HeadBias      bool   `json:"head_bias"`
}

type LayerArray struct {
	bufferStart   int
	rechannel     *conv1x1.Model
	layerBuffers  []mat.Matrix
	layers        []*layer.Layer
	headRechannel *conv1x1.Model
}

const layerArrayBufferSize = 65536

func NewLayerArray(config Config) *LayerArray {
	la := &LayerArray{
		rechannel: conv1x1.New(conv1x1.Config{
			InChannels:  config.InputSize,
			OutChannels: config.Channels,
			Bias:        false,
		}),
		layerBuffers: make([]mat.Matrix, len(config.Dilations)),
		layers:       make([]*layer.Layer, len(config.Dilations)),
		headRechannel: conv1x1.New(conv1x1.Config{
			InChannels:  config.Channels,
			OutChannels: config.HeadSize,
			Bias:        config.HeadBias,
		}),
	}

	for i, dilation := range config.Dilations {
		la.layers[i] = layer.New(layer.Config{
			ConditionSize: config.ConditionSize,
			Channels:      config.Channels,
			KernelSize:    config.KernelSize,
			Dilation:      dilation,
			Activation:    config.Activation,
			Gated:         config.Gated,
		})
	}

	receptiveField := la.GetReceptiveField()
	layerColumns := layerArrayBufferSize + receptiveField
	for i := range config.Dilations {
		la.layerBuffers[i] = mat.NewMatrix(config.Channels, layerColumns)
	}

	la.bufferStart = receptiveField
	return la
}

func (la *LayerArray) AdvanceBuffers(numFrames int) {
	la.bufferStart += numFrames
}

// GetReceptiveField returns the zero-indexed receptive field.
func (la *LayerArray) GetReceptiveField() int {
	receptiveField := 0
	for _, l := range la.layers {
		receptiveField += (l.GetKernelSize() - 1) * l.GetDilation()
	}
	return receptiveField
}

func (la *LayerArray) PrepareForFrames(numFrames int) {
	if la.bufferStart+numFrames > la.getBufferSize() {
		la.rewindBuffers()
	}
}

func (la *LayerArray) getBufferSize() int {
	if len(la.layerBuffers) == 0 {
		return 0
	}
	return la.layerBuffers[0].Columns()
}

func (la *LayerArray) getChannels() int {
	if len(la.layers) == 0 {
		return 0
	}
	return la.layers[0].GetChannels()
}

func (la *LayerArray) rewindBuffers() {
	start := la.GetReceptiveField()

	for i, layerBuffer := range la.layerBuffers {
		l := la.layers[i]
		d := (l.GetKernelSize() - 1) * l.GetDilation()
		mat.Copy(
			layerBuffer.ViewMiddleColumns(start-d, d),
			layerBuffer.ViewMiddleColumns(la.bufferStart-d, d),
		)
	}
	la.bufferStart = start
}

func (la *LayerArray) SetParams(params *floatreader.Reader) {
	la.rechannel.SetParams(params)
	for _, l := range la.layers {
		l.SetParams(params)
	}
	la.headRechannel.SetParams(params)
}

func (la *LayerArray) SetNumFrames(numFrames int) {
	if layerArrayBufferSize-numFrames <= la.GetReceptiveField() {
		panic("buffer is too short")
	}
	for _, l := range la.layers {
		l.SetNumFrames(numFrames)
	}
}

func (la *LayerArray) Process(layerInputs, condition, headInputs, layerOutputs, headOutputs mat.Matrix) {
	la.rechannel.Process(
		layerInputs,
		la.layerBuffers[0].ViewMiddleColumns(la.bufferStart, layerInputs.Columns()),
	)

	lastIndex := len(la.layers) - 1
	for i, l := range la.layers[:lastIndex] {
		l.Process(
			la.layerBuffers[i],
			condition,
			headInputs,
			la.layerBuffers[i+1],
			la.bufferStart,
			la.bufferStart,
		)
	}
	la.layers[lastIndex].Process(
		la.layerBuffers[lastIndex],
		condition,
		headInputs,
		layerOutputs,
		la.bufferStart,
		0,
	)

	la.headRechannel.Process(headInputs, headOutputs)
}
