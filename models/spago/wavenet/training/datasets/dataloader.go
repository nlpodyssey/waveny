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

package datasets

import (
	"fmt"
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
)

type DataLoaderConfig struct {
	BatchSize int
	Shuffle   bool
	DropLast  bool
}

type DataLoader struct {
	dataset      *Dataset
	batchSize    int
	dropLast     bool
	sampler      Sampler[int]
	batchSampler *BatchSampler
	length       int
}

func NewDataLoader(dataset *Dataset, config DataLoaderConfig) *DataLoader {
	if config.BatchSize <= 0 {
		panic(fmt.Errorf("invalid or unsupported batch size %d", config.BatchSize))
	}

	var sampler Sampler[int]
	if config.Shuffle {
		sampler = NewRandomSampler(dataset)
	} else {
		sampler = NewSequentialSampler(dataset)
	}

	batchSampler := NewBatchSampler(sampler, config.BatchSize, config.DropLast)

	return &DataLoader{
		dataset:      dataset,
		batchSize:    config.BatchSize,
		dropLast:     config.DropLast,
		sampler:      sampler,
		batchSampler: batchSampler,
		length:       batchSampler.Length(),
	}
}

type XYDataPair struct {
	X mat.Tensor
	Y mat.Tensor
}

func (dl *DataLoader) Iterate(f func(XYDataPair)) {
	xs := make([]mat.Tensor, dl.batchSize)
	ys := make([]mat.Tensor, dl.batchSize)
	dl.batchSampler.Iterate(func(batchIndices []int) {
		for i, index := range batchIndices {
			xs[i], ys[i] = dl.dataset.GetItem(index)
		}
		f(collate(xs, ys))
	})
}

func collate(xs, ys []mat.Tensor) XYDataPair {
	return XYDataPair{
		X: ag.Stack(xs...),
		Y: ag.Stack(ys...),
	}
}
