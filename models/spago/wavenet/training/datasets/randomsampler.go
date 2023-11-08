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
	"math/rand"
	"time"
)

type RandomSampler struct {
	dataSource *Dataset
	generator  *rand.Rand
}

func NewRandomSampler(dataSource *Dataset) *RandomSampler {
	return &RandomSampler{
		dataSource: dataSource,
		generator:  rand.New(rand.NewSource(time.Now().UnixNano() + rand.Int63())),
	}
}

func (rs *RandomSampler) Length() int {
	return rs.dataSource.Length()
}

func (rs *RandomSampler) Iterate(f func(int)) {
	n := rs.dataSource.Length()
	perm := rs.generator.Perm(n)
	for _, v := range perm {
		f(v)
	}
}
