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
	"github.com/nlpodyssey/waveny/models/realtime/activations"
	"github.com/nlpodyssey/waveny/models/realtime/conv1d"
	"github.com/nlpodyssey/waveny/models/realtime/conv1x1"
	"github.com/nlpodyssey/waveny/models/realtime/floatreader"
	"github.com/nlpodyssey/waveny/models/realtime/mat"
)

type Config struct {
	ConditionSize int
	Channels      int
	KernelSize    int
	Dilation      int
	Activation    string
	Gated         bool
}

type Layer struct {
	frontConv  *conv1d.Model
	inputMixin *conv1x1.Model
	postConv   *conv1x1.Model
	activation activations.Activation
	gated      bool
	state      mat.Matrix
	tmpState   mat.Matrix
}

func New(config Config) *Layer {
	if config.Gated {
		panic("gated layer is not implemented")
	}

	outChannels := config.Channels
	if config.Gated {
		outChannels *= 2
	}
	return &Layer{
		frontConv: conv1d.New(conv1d.Config{
			InChannels:  config.Channels,
			OutChannels: outChannels,
			KernelSize:  config.KernelSize,
			Bias:        true,
			Dilation:    config.Dilation,
		}),
		inputMixin: conv1x1.New(conv1x1.Config{
			InChannels:  config.ConditionSize,
			OutChannels: outChannels,
			Bias:        false,
		}),
		postConv: conv1x1.New(conv1x1.Config{
			InChannels:  config.Channels,
			OutChannels: config.Channels,
			Bias:        true,
		}),
		activation: activations.New(config.Activation),
		gated:      config.Gated,
	}
}

func (l *Layer) SetNumFrames(numFrames int) {
	convOutChannels := l.frontConv.GetOutChannels()
	if l.state.Rows() == convOutChannels && l.state.Columns() == numFrames {
		return
	}

	l.state = l.state.Resize(convOutChannels, numFrames)
	l.state.SetZero()

	l.tmpState = l.tmpState.Resize(convOutChannels, numFrames)
}

func (l *Layer) GetChannels() int {
	return l.frontConv.GetInChannels()
}

func (l *Layer) GetDilation() int {
	return l.frontConv.GetDilation()
}

func (l *Layer) GetKernelSize() int {
	return l.frontConv.GetKernelSize()
}

func (l *Layer) SetParams(params *floatreader.Reader) {
	l.frontConv.SetParams(params)
	l.inputMixin.SetParams(params)
	l.postConv.SetParams(params)
}

func (l *Layer) Process(input, condition, headInput, output mat.Matrix, inputStartColumn, outputStartColumn int) {
	numColumns := condition.Columns()
	channels := l.GetChannels()

	l.frontConv.Process(input, l.state, inputStartColumn, numColumns, 0)

	l.inputMixin.Process(condition, l.tmpState)
	mat.AddInPlace(l.state, l.tmpState)

	l.activation.Apply(l.state)

	// TODO: implement gated
	//if l.gated {
	//	activations::Activation::get_activation("Sigmoid")->apply(this->_z.block(channels, 0, channels, this->_z.cols()));
	//
	//	this->_z.topRows(channels).array() *= this->_z.bottomRows(channels).array();
	//	// this->_z.topRows(channels) = this->_z.topRows(channels).cwiseProduct(
	//	//   this->_z.bottomRows(channels)
	//	// );
	//}

	topState := l.state.ViewTopRows(channels)
	mat.AddInPlace(headInput, topState)

	outputView := output.ViewMiddleColumns(outputStartColumn, numColumns)
	l.postConv.Process(topState, outputView)
	mat.AddInPlace(outputView, input.ViewMiddleColumns(inputStartColumn, numColumns))
}
