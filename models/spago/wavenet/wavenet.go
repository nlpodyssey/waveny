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
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/losses"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/waveny/models/spago/wavenet/layerarray"
	"github.com/nlpodyssey/waveny/models/spago/wavenet/training/datasets"
)

// A Config specifies the configuration for instantiating a new WaveNet Model.
// The fields are matching the JSON model configuration.
type Config struct {
	HeadScale     float32             `json:"head_scale"`
	LayersConfigs []layerarray.Config `json:"layers_configs"`
}

type Model struct {
	nn.Module
	Layers         []*layerarray.Model
	HeadScale      mat.Tensor
	ReceptiveField int
	// Avoid recomputing padding vector at each inference
	ZeroPadding mat.Tensor
	// TODO: WaveNet Head not implemented
}

// New creates a new WaveNet Model.
func New(config Config) *Model {
	layers := makeLayers(config.LayersConfigs)
	receptiveField := computeReceptiveField(layers)
	return &Model{
		Layers:         layers,
		HeadScale:      ag.StopGrad(mat.Scalar(config.HeadScale)),
		ReceptiveField: receptiveField,
		ZeroPadding:    ag.StopGrad(mat.NewDense[float32](mat.WithShape(1, receptiveField-1))),
	}
}

func makeLayers(configs []layerarray.Config) []*layerarray.Model {
	layers := make([]*layerarray.Model, len(configs))
	for i, layerArrayConfig := range configs {
		layers[i] = layerarray.New(layerArrayConfig)
	}
	return layers
}

func computeReceptiveField(layers []*layerarray.Model) int {
	s := 1
	for _, l := range layers {
		s += l.ReceptiveField - 1
	}
	return s
}

func (m *Model) Forward(x mat.Tensor, padStart bool) mat.Tensor {
	if mat.IsVector(x) {
		return m.forwardOne(x, padStart)
	}
	return ag.Stack(ag.Map(
		func(t mat.Tensor) mat.Tensor {
			return m.forwardOne(t, padStart)
		},
		ag.RowViews(x),
	)...)
}

func (m *Model) forwardOne(x mat.Tensor, padStart bool) mat.Tensor {
	if padStart {
		x = ag.Concat(m.ZeroPadding, x)
	}
	if x.Shape()[0] != 1 { // Always work with a row vector
		x = ag.T(x)
	}

	y := x
	headInput := mat.Tensor(nil)
	for _, layers := range m.Layers {
		headInput, y = layers.Forward(y, x, headInput)
	}

	return ag.ProdScalar(headInput, m.HeadScale)
}

func (m *Model) ResetParameters() {
	for _, layers := range m.Layers {
		layers.ResetParameters()
	}
}

func (m *Model) TrainingStep(batch datasets.XYDataPair) mat.Tensor {
	preds := m.Forward(batch.X, false)
	return losses.MSE(preds, batch.Y, true)
}

func (m *Model) ValidationStep(batch datasets.XYDataPair) (mseLoss, esrLoss float32) {
	preds := m.Forward(batch.X, false)
	mse := losses.MSE(preds, batch.Y, true)
	esr := computeESRLoss(preds, batch.Y)
	return mse.Item().F32(), esr.Item().F32()
}

// computeESRLoss computes the Error Signal Ratio (ESR) loss.
func computeESRLoss(preds, targets mat.Tensor) mat.Tensor {
	return ag.ReduceMean(
		ag.Div(
			ag.ReduceMean(ag.Square(ag.Sub(preds, targets))),
			ag.ReduceMean(ag.Square(targets)),
		),
	)
}
