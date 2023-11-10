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
	"strings"
	"testing"
)

func TestNewMatrixFromSlices(t *testing.T) {
	testCases := []struct {
		name     string
		data     [][]float32
		expected Matrix
	}{
		{
			"nil",
			nil,
			Matrix{},
		},
		{
			"empty",
			[][]float32{},
			Matrix{},
		},
		{
			"one empty row",
			[][]float32{{}},
			Matrix{
				columns:     0,
				dataRows:    1,
				viewFromRow: 0,
				viewRows:    1,
				data:        []float32{},
			},
		},
		{
			"two empty rows",
			[][]float32{{}, {}},
			Matrix{
				columns:     0,
				dataRows:    2,
				viewFromRow: 0,
				viewRows:    2,
				data:        []float32{},
			},
		},
		{
			"scalar",
			[][]float32{{42}},
			Matrix{
				columns:     1,
				dataRows:    1,
				viewFromRow: 0,
				viewRows:    1,
				data:        []float32{42},
			},
		},
		{
			"1x2",
			[][]float32{{1, 2}},
			Matrix{
				columns:     2,
				dataRows:    1,
				viewFromRow: 0,
				viewRows:    1,
				data:        []float32{1, 2},
			},
		},
		{
			"2x1",
			[][]float32{{1}, {2}},
			Matrix{
				columns:     1,
				dataRows:    2,
				viewFromRow: 0,
				viewRows:    2,
				data:        []float32{1, 2},
			},
		},
		{
			"2x3",
			[][]float32{{1, 2, 3}, {4, 5, 6}},
			Matrix{
				columns:     3,
				dataRows:    2,
				viewFromRow: 0,
				viewRows:    2,
				data:        []float32{1, 4, 2, 5, 3, 6},
			},
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			actual := NewMatrixFromSlices(tc.data)
			requireDeepEqual(t, tc.expected, actual)
		})
	}
}

func TestMatrix_Rows_Columns(t *testing.T) {
	v := NewMatrix(3, 4)
	if v.Rows() != 3 {
		t.Errorf("expected 3 rows, actual %d", v.Rows())
	}
	if v.Columns() != 4 {
		t.Errorf("expected 4 columns, actual %d", v.Columns())
	}
}
func TestMatrix_Set_Get(t *testing.T) {
	m := NewMatrix(3, 2)

	m.Set(0, 0, 100)
	m.Set(0, 1, 101)

	m.Set(1, 0, 110)
	m.Set(1, 1, 111)

	m.Set(2, 0, 120)
	m.Set(2, 1, 121)

	requireFloat32Equal(t, 100, m.Get(0, 0))
	requireFloat32Equal(t, 101, m.Get(0, 1))

	requireFloat32Equal(t, 110, m.Get(1, 0))
	requireFloat32Equal(t, 111, m.Get(1, 1))

	requireFloat32Equal(t, 120, m.Get(2, 0))
	requireFloat32Equal(t, 121, m.Get(2, 1))
}

func TestMatrix_Clone(t *testing.T) {
	m := Matrix{}
	requireDeepEqual(t, m, m.Clone())

	m = NewMatrixFromSlices([][]float32{
		{100, 101, 102, 103},
		{110, 111, 112, 113},
		{120, 121, 122, 123},
		{130, 131, 132, 133},
	})
	requireDeepEqual(t, m, m.Clone())

	view := m.View(1, 1, 2, 2)
	viewClone := view.Clone()
	requireDeepEqual(t, Matrix{
		columns:     2,
		dataRows:    2,
		viewFromRow: 0,
		viewRows:    2,
		data: []float32{
			111, 121,
			112, 122,
		},
	}, viewClone)
	assertMatrixEqual(t, NewMatrixFromSlices([][]float32{
		{111, 112},
		{121, 122},
	}), viewClone)
}

func TestMatrix_AsVector(t *testing.T) {
	m := NewMatrix(3, 1)
	v := m.AsVector()
	requireDeepEqual(t, m, v.Matrix)
	if &v.data[0] != &m.data[0] {
		t.Fatalf("same data expected")
	}
}

func TestMatrix_Resize(t *testing.T) {
	m := NewMatrixFromSlices([][]float32{
		{1, 2, 3},
		{4, 5, 6},
	})

	a := m.Resize(2, 3)
	if &a.data[0] != &m.data[0] {
		t.Fatalf("same data expected")
	}

	b := m.Resize(3, 2)
	if &b.data[0] == &m.data[0] {
		t.Fatalf("different data expected")
	}
}

func TestMatrix_String(t *testing.T) {
	m := NewMatrixFromSlices([][]float32{
		{1, 2, 3},
		{4, 5, 6},
	})
	actual := m.String()
	expected := strings.Join([]string{
		"Matrix(2x3)[",
		"  [ 1 2 3 ]",
		"  [ 4 5 6 ]",
		"]",
	}, "\n")
	if actual != expected {
		t.Fatalf("expected:\n%q\nactual:\n%q", expected, actual)
	}
}

func requireFloat32Equal(t *testing.T, expected, actual float32) {
	t.Helper()
	if math.Abs(float64(expected-actual)) > 1e-5 {
		t.Fatalf("values differ\nexpected:\n%#v\nactual:\n%#v", expected, actual)
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
