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

type BatchSampler struct {
	sampler   Sampler[int]
	batchSize int
	dropLast  bool
}

func NewBatchSampler(sampler Sampler[int], batchSize int, dropLast bool) *BatchSampler {
	return &BatchSampler{
		sampler:   sampler,
		batchSize: batchSize,
		dropLast:  dropLast,
	}
}

func (bs *BatchSampler) Length() int {
	if bs.dropLast {
		return bs.sampler.Length() / bs.batchSize
	}
	return (bs.sampler.Length() + bs.batchSize - 1) / bs.batchSize
}

func (bs *BatchSampler) Iterate(f func([]int)) {
	if bs.dropLast {
		batchIndices := make([]int, 0, bs.batchSize)
		bs.sampler.Iterate(func(v int) {
			batchIndices = append(batchIndices, v)
			if len(batchIndices) == bs.batchSize {
				f(batchIndices)
				batchIndices = batchIndices[:0]
			}
		})
		return
	}

	batch := make([]int, bs.batchSize)
	idxInBatch := 0
	bs.sampler.Iterate(func(idx int) {
		batch[idxInBatch] = idx
		idxInBatch += 1
		if idxInBatch == bs.batchSize {
			f(batch)
			idxInBatch = 0
			for i := range batch {
				batch[i] = 0
			}
		}
	})
	if idxInBatch > 0 {
		f(batch[:idxInBatch])
	}
}
