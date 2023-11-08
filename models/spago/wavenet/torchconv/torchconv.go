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

package torchconv

import (
	"errors"
	"fmt"
	"github.com/nlpodyssey/gopickle/pytorch"
	"github.com/nlpodyssey/gopickle/types"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/waveny/models/spago/conv1d"
	"github.com/nlpodyssey/waveny/models/spago/conv1x1"
	"github.com/nlpodyssey/waveny/models/spago/wavenet"
	"github.com/nlpodyssey/waveny/models/spago/wavenet/layer"
	"github.com/nlpodyssey/waveny/models/spago/wavenet/layerarray"
	"strings"
)

func LoadTorchModel(checkpointPath string, model *wavenet.Model) error {
	checkpoint, err := pytorch.Load(checkpointPath)
	if err != nil {
		return err
	}
	stateDict, err := getStateDic(checkpoint)
	if err != nil {
		return err
	}
	state, err := makeStateMap(stateDict)
	if err != nil {
		return err
	}
	return loadModelState(model, state)
}

func loadModelState(model *wavenet.Model, state StateMap) error {
	for i, layerArray := range model.Layers {
		layerArrayState := state.ExtractPrefixedSubset(fmt.Sprintf("_net._net._layers.%d.", i))
		if err := loadModelLayerArray(layerArray, layerArrayState); err != nil {
			return fmt.Errorf("failed to load state for layer-array %d: %w", i, err)
		}
	}
	return nil
}

func loadModelLayerArray(model *layerarray.Model, state StateMap) error {
	if err := loadConv1D(model.Rechannel, state.ExtractPrefixedSubset("_rechannel.")); err != nil {
		return fmt.Errorf("failed to load Rechannel: %w", err)
	}
	if err := loadConv1D(model.HeadRechannel, state.ExtractPrefixedSubset("_head_rechannel.")); err != nil {
		return fmt.Errorf("failed to load HeadRechannel: %w", err)
	}
	for i, l := range model.Layers {
		layerState := state.ExtractPrefixedSubset(fmt.Sprintf("_layers.%d.", i))
		if err := loadModelLayer(l, layerState); err != nil {
			return fmt.Errorf("failed to load state for sub-layer %d: %w", i, err)
		}
	}
	return nil
}

func loadModelLayer(model *layer.Model, state StateMap) error {
	if err := loadConv1D(model.Conv, state.ExtractPrefixedSubset("_conv.")); err != nil {
		return fmt.Errorf("failed to load layer Conv: %w", err)
	}
	if err := loadConv1D(model.InputMixer, state.ExtractPrefixedSubset("_input_mixer.")); err != nil {
		return fmt.Errorf("failed to load layer InputMixer: %w", err)
	}
	if err := loadConv1x1(model.Conv1x1, state.ExtractPrefixedSubset("_1x1.")); err != nil {
		return fmt.Errorf("failed to load layer Conv1x1: %w", err)
	}
	return nil
}

func loadConv1D(model *conv1d.Model, state StateMap) error {
	if err := loadConv1DWeight(model, state); err != nil {
		return err
	}
	if model.Bias != nil {
		return loadConv1DBias(model, state)
	}
	return nil
}

func loadConv1DWeight(model *conv1d.Model, state StateMap) error {
	weightTensor, ok := state["weight"]
	if !ok {
		return errors.New("conv1d weight not found")
	}
	rawWeight, err := getRawTensorData(weightTensor)
	if err != nil {
		return fmt.Errorf("failed to get conv1d weight data: %w", err)
	}

	outChannels := model.OutChannels
	inChannels := model.InChannels
	weights := model.Weights
	p := 0
	for i := 0; i < outChannels; i++ {
		for j := 0; j < inChannels; j++ {
			for k := range weights {
				weights[k].SetAt(mat.Scalar(rawWeight[p]), i, j)
				p += 1
			}
		}
	}
	return nil
}

func loadConv1DBias(model *conv1d.Model, state StateMap) error {
	biasTensor, ok := state["bias"]
	if !ok {
		return errors.New("conv1d bias not found")
	}
	rawBias, err := getRawTensorData(biasTensor)
	if err != nil {
		return fmt.Errorf("failed to get conv1d bias data: %w", err)
	}
	model.Bias.SetData(float.Make(rawBias...))
	return nil
}

func loadConv1x1(model *conv1x1.Model, state StateMap) error {
	if err := loadConv1x1Weight(model, state); err != nil {
		return err
	}
	return loadConv1x1Bias(model, state)
}

func loadConv1x1Weight(model *conv1x1.Model, state StateMap) error {
	weightTensor, ok := state["weight"]
	if !ok {
		return errors.New("conv1x1 weight not found")
	}
	rawWeight, err := getRawTensorData(weightTensor)
	if err != nil {
		return fmt.Errorf("failed to get conv1x1 weight data: %w", err)
	}
	model.Weights.SetData(float.Make(rawWeight...))
	return nil
}

func loadConv1x1Bias(model *conv1x1.Model, state StateMap) error {
	biasTensor, ok := state["bias"]
	if !ok {
		return errors.New("conv1x1 bias not found")
	}
	rawBias, err := getRawTensorData(biasTensor)
	if err != nil {
		return fmt.Errorf("failed to get conv1x1 bias data: %w", err)
	}
	model.Bias.SetData(float.Make(rawBias...))
	return nil
}

type StateMap map[string]*pytorch.Tensor

func (s StateMap) ExtractPrefixedSubset(prefix string) StateMap {
	sub := make(StateMap, len(s))
	for key, value := range s {
		if after, found := strings.CutPrefix(key, prefix); found {
			sub[after] = value
		}
	}
	return sub
}

func makeStateMap(stateDict *types.OrderedDict) (StateMap, error) {
	m := make(StateMap, stateDict.Len())
	for rawKey, entry := range stateDict.Map {
		key, err := cast[string](rawKey)
		if err != nil {
			return nil, fmt.Errorf(`failed to cast state-dict's key to string: %w`, err)
		}
		value, err := cast[*pytorch.Tensor](entry.Value)
		if err != nil {
			return nil, fmt.Errorf(`failed to cast state-dict's %q value to Tensor: %w`, key, err)
		}
		m[key] = value
	}
	return m, nil
}

func getStateDic(checkpoint any) (*types.OrderedDict, error) {
	checkpointDict, err := cast[*types.Dict](checkpoint)
	if err != nil {
		return nil, fmt.Errorf("failed to cast checkpoint to Dict: %w", err)
	}
	rawStateDict, ok := checkpointDict.Get("state_dict")
	if !ok {
		return nil, errors.New(`"state_dict" not found`)
	}
	stateDict, err := cast[*types.OrderedDict](rawStateDict)
	if err != nil {
		return nil, fmt.Errorf(`failed to cast "state_dict" to OrderedDict: %w`, err)
	}
	return stateDict, nil
}

func getRawTensorData(t *pytorch.Tensor) ([]float32, error) {
	if t.StorageOffset != 0 {
		return nil, fmt.Errorf("only 0 storage-offset is supported, actual: %v", t.StorageOffset)
	}
	source, err := cast[*pytorch.FloatStorage](t.Source)
	if err != nil {
		return nil, fmt.Errorf("unsupported storage: %w", err)
	}
	tensorSize := 1
	for _, v := range t.Size {
		tensorSize *= v
	}
	if tensorSize != source.Size {
		return nil, fmt.Errorf("tensor total size %d is incompatible with source storage size %d", tensorSize, source.Size)
	}
	if tensorSize != len(source.Data) {
		return nil, fmt.Errorf("tensor total size %d is incompatible with source storage data length %d", tensorSize, len(source.Data))
	}
	return source.Data, nil
}

func cast[T any](v any) (T, error) {
	c, ok := v.(T)
	if !ok {
		return c, fmt.Errorf("expected type %T, actual %T", c, v)
	}
	return c, nil
}
