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
	"fmt"
	"github.com/nlpodyssey/waveny/models/realtime/floatreader"
	"github.com/nlpodyssey/waveny/models/realtime/mat"
	"github.com/nlpodyssey/waveny/models/realtime/wavenet/layerarray"
)

type Config struct {
	HeadScale float32             `json:"head_scale"`
	Head      *any                `json:"head"`
	Layers    []layerarray.Config `json:"layers"`
}

type Model struct {
	numFrames         int
	layerArrays       []*layerarray.LayerArray
	layerArrayOutputs []mat.Matrix
	condition         mat.Matrix
	headArrays        []mat.Matrix
	headScale         float32
	headOutput        mat.Matrix
}

func New(config Config, params *floatreader.Reader) (*Model, error) {
	if config.Head != nil {
		return nil, fmt.Errorf("head not implemented")
	}
	if len(config.Layers) < 2 {
		return nil, fmt.Errorf("expected at least two layers, actual %d", len(config.Layers))
	}

	wn := &Model{
		numFrames:         0,
		layerArrays:       make([]*layerarray.LayerArray, len(config.Layers)),
		layerArrayOutputs: make([]mat.Matrix, len(config.Layers)),
		headArrays:        make([]mat.Matrix, 1+len(config.Layers)),
		headScale:         config.HeadScale,
		headOutput:        mat.NewMatrix(1, 0),
	}

	wn.headArrays[0] = mat.NewMatrix(config.Layers[0].Channels, 0)

	for i, layerArrayConfig := range config.Layers {
		wn.layerArrays[i] = layerarray.NewLayerArray(layerArrayConfig)
		wn.layerArrayOutputs[i] = mat.NewMatrix(layerArrayConfig.Channels, 0)

		if i > 0 && layerArrayConfig.Channels != config.Layers[i-1].HeadSize {
			return nil, fmt.Errorf(
				"channels of layer %d (%d) don't match head size of previous layer (%d)",
				i, layerArrayConfig.Channels, config.Layers[i-1].HeadSize)
		}
		wn.headArrays[i+1] = mat.NewMatrix(layerArrayConfig.HeadSize, 0)
	}

	if err := wn.SetParams(params); err != nil {
		return nil, err
	}

	wn.warmUp()
	return wn, nil
}

func (m *Model) warmUp() {
	receptiveField := m.getReceptiveField()
	samples := []float32{0}
	for i := 0; i < receptiveField; i++ {
		m.Process(samples, samples)
		m.Finalize(1)
		samples[0] = 0
	}
}

func (m *Model) getReceptiveField() int {
	receptiveField := 1
	for _, layerArray := range m.layerArrays {
		receptiveField += layerArray.GetReceptiveField()
	}
	return receptiveField
}

func (m *Model) Finalize(numFrames int) {
	m.advanceBuffers(numFrames)
}

func (m *Model) SetParams(params *floatreader.Reader) error {
	for _, layerArray := range m.layerArrays {
		layerArray.SetParams(params)
	}
	m.headScale = params.Next()
	if params.HasNext() {
		return fmt.Errorf("too many parameters")
	}
	return nil
}

func (m *Model) advanceBuffers(numFrames int) {
	for _, layerArray := range m.layerArrays {
		layerArray.AdvanceBuffers(numFrames)
	}
}

func (m *Model) prepareForFrames(numFrames int) {
	for _, layerArray := range m.layerArrays {
		layerArray.PrepareForFrames(numFrames)
	}
}

func (m *Model) setNumFrames(numFrames int) {
	if numFrames == m.numFrames {
		return
	}

	m.condition = m.condition.Resize(1, numFrames)
	for i, headArray := range m.headArrays {
		m.headArrays[i] = headArray.Resize(headArray.Rows(), numFrames)
	}
	for i, layerArrayOutput := range m.layerArrayOutputs {
		m.layerArrayOutputs[i] = layerArrayOutput.Resize(layerArrayOutput.Rows(), numFrames)
	}

	m.headOutput = m.headOutput.Resize(m.headOutput.Rows(), numFrames)
	m.headOutput.SetZero()

	for _, layerArray := range m.layerArrays {
		layerArray.SetNumFrames(numFrames)
	}
	m.numFrames = numFrames
}

func (m *Model) Process(input, output []float32) {
	numFrames := len(input)
	m.setNumFrames(numFrames)
	m.prepareForFrames(numFrames)

	for j, inputValue := range input {
		m.condition.Set(0, j, inputValue)
	}

	m.headArrays[0].SetZero()

	m.layerArrays[0].Process(
		m.condition,
		m.condition,
		m.headArrays[0],
		m.layerArrayOutputs[0],
		m.headArrays[1],
	)
	for i := 1; i < len(m.layerArrays); i++ {
		m.layerArrays[i].Process(
			m.layerArrayOutputs[i-1],
			m.condition,
			m.headArrays[i],
			m.layerArrayOutputs[i],
			m.headArrays[i+1],
		)
	}

	finalHeadArray := m.headArrays[len(m.headArrays)-1]
	for i := range output {
		output[i] = m.headScale * finalHeadArray.Get(0, i)
	}
}
