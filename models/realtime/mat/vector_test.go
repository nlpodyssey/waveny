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

import (
	"testing"
)

func TestNewVectorFromSlice(t *testing.T) {
	testCases := []struct {
		name     string
		actual   Vector
		expected Vector
	}{
		{
			"nil slice",
			NewVectorFromSlice(nil),
			Vector{},
		},
		{
			"empty slice",
			NewVectorFromSlice([]float32{}),
			Vector{},
		},
		{
			"slice with data",
			NewVectorFromSlice([]float32{1, 2, 3}),
			Vector{Matrix{
				rows:        3,
				dataColumns: 1,
				viewColumns: 1,
				data:        []float32{1, 2, 3},
			}},
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			assertMatrixEqual(t, tc.expected.Matrix, tc.actual.Matrix)
		})
	}
}
