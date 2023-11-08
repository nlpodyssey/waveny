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

package mat

type Vector struct {
	Matrix
}

func NewVector(size int) Vector {
	return Vector{Matrix: NewMatrix(size, 1)}
}

func NewVectorFromSlice(data []float32) Vector {
	size := len(data)
	if size == 0 {
		return Vector{}
	}
	v := NewVector(size)
	copy(v.data, data)
	return v
}

func (v Vector) Size() int {
	return v.Rows() * v.Columns()
}

func (v Vector) Set(index int, value float32) {
	v.Matrix.Set(index, 0, value)
}

func (v Vector) Get(index int) float32 {
	return v.Matrix.Get(index, 0)
}
