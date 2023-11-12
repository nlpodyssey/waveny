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
	"math"
	"reflect"
	"testing"
)

func TestMatrix_Clone(t *testing.T) {
	m := NewMatrixFromSlices([][]float32{
		{100, 101, 102, 103},
		{110, 111, 112, 113},
		{120, 121, 122, 123},
		{130, 131, 132, 133},
	})

	requireDeepEqual(t, m, m.Clone())

	view := m.View(1, 1, 2, 2)
	viewClone := view.Clone()
	requireDeepEqual(t, Matrix{
		rows:        2,
		dataColumns: 2,
		viewColumns: 2,
		data: []float32{
			111, 112,
			121, 122,
		},
		scratch: make([]float32, 4),
	}, viewClone)
	assertMatrixEqual(t, NewMatrixFromSlices([][]float32{
		{111, 112},
		{121, 122},
	}), viewClone)

}

func TestNewMatrixFromSlices(t *testing.T) {
	testCases := []struct {
		name     string
		actual   Matrix
		expected Matrix
	}{
		{
			"nil slice",
			NewMatrixFromSlices(nil),
			Matrix{},
		},
		{
			"empty slice",
			NewMatrixFromSlices([][]float32{}),
			Matrix{},
		},
		{
			"slice with data",
			NewMatrixFromSlices([][]float32{
				{1, 2, 3},
				{4, 5, 6},
			}),
			Matrix{
				rows:        2,
				dataColumns: 3,
				viewColumns: 3,
				data:        []float32{1, 2, 3, 4, 5, 6},
			},
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			assertMatrixEqual(t, tc.expected, tc.actual)
		})
	}
}

func requireDeepEqual(t *testing.T, expected, actual any) {
	t.Helper()
	if !reflect.DeepEqual(expected, actual) {
		t.Fatalf("values differ\nexpected:\n%#v\nactual:\n%#v", expected, actual)
	}
}

func assertMatrixEqual(t *testing.T, expected, actual Matrix) {
	t.Helper()
	if expected.Rows() != actual.Rows() || expected.Columns() != actual.Columns() {
		t.Errorf("different shapes\nexpected:\n%v\nactual:\n%v", expected, actual)
		return
	}
	rows := expected.Rows()
	columns := expected.Columns()
	for i := 0; i < rows; i++ {
		for j := 0; j < columns; j++ {
			e := expected.Get(i, j)
			a := actual.Get(i, j)
			if math.Abs(float64(e-a)) > 1e-5 {
				t.Errorf("different values (first mismatch at %dx%d)\nexpected:\n%v\nactual:\n%v", i, j, expected, actual)
				return
			}
		}
	}
}
