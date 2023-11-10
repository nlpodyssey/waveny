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

import "testing"

func TestNewVectorFromSlice(t *testing.T) {
	testCases := []struct {
		name     string
		data     []float32
		expected Vector
	}{
		{
			"nil",
			nil,
			Vector{},
		},
		{
			"empty",
			[]float32{},
			Vector{},
		},
		{
			"scalar",
			[]float32{42},
			Vector{Matrix{
				columns:     1,
				dataRows:    1,
				viewFromRow: 0,
				viewRows:    1,
				data:        []float32{42},
			}},
		},
		{
			"size 3",
			[]float32{1, 2, 3},
			Vector{Matrix{
				columns:     1,
				dataRows:    3,
				viewFromRow: 0,
				viewRows:    3,
				data:        []float32{1, 2, 3},
			}},
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			actual := NewVectorFromSlice(tc.data)
			requireDeepEqual(t, tc.expected, actual)
		})
	}
}

func TestVector_Size(t *testing.T) {
	v := NewVector(3)
	if v.Size() != 3 {
		t.Errorf("expected 3, actual %d", v.Size())
	}
}

func TestVector_Set_Get(t *testing.T) {
	v := NewVector(3)

	v.Set(0, 100)
	v.Set(1, 101)
	v.Set(2, 102)

	requireFloat32Equal(t, 100, v.Get(0))
	requireFloat32Equal(t, 101, v.Get(1))
	requireFloat32Equal(t, 102, v.Get(2))
}
