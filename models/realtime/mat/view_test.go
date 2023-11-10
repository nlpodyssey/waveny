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

func TestMatrix_View(t *testing.T) {
	m := NewMatrixFromSlices([][]float32{
		{100, 101, 102, 103},
		{110, 111, 112, 113},
		{120, 121, 122, 123},
	})

	requireDeepEqual(t, Matrix{
		columns:     4,
		dataRows:    3,
		viewFromRow: 0,
		viewRows:    3,
		data: []float32{
			100, 110, 120,
			101, 111, 121,
			102, 112, 122,
			103, 113, 123,
		},
	}, m)

	vIdentity := m.View(0, 0, 3, 4)
	requireDeepEqual(t, m, vIdentity)

	vTop := m.View(0, 0, 2, 4)
	requireDeepEqual(t, Matrix{
		columns:     4,
		dataRows:    3,
		viewFromRow: 0,
		viewRows:    2,
		data:        m.data,
	}, vTop)
	assertMatrixEqual(t, NewMatrixFromSlices([][]float32{
		{100, 101, 102, 103},
		{110, 111, 112, 113},
	}), vTop)

	vBottom := m.View(1, 0, 2, 4)
	requireDeepEqual(t, Matrix{
		columns:     4,
		dataRows:    3,
		viewFromRow: 1,
		viewRows:    2,
		data:        m.data,
	}, vBottom)
	assertMatrixEqual(t, NewMatrixFromSlices([][]float32{
		{110, 111, 112, 113},
		{120, 121, 122, 123},
	}), vBottom)

	vLeft := m.View(0, 0, 3, 2)
	requireDeepEqual(t, Matrix{
		columns:     2,
		dataRows:    3,
		viewFromRow: 0,
		viewRows:    3,
		data: []float32{
			100, 110, 120,
			101, 111, 121,
		},
	}, vLeft)
	assertMatrixEqual(t, NewMatrixFromSlices([][]float32{
		{100, 101},
		{110, 111},
		{120, 121},
	}), vLeft)

	vRight := m.View(0, 2, 3, 2)
	requireDeepEqual(t, Matrix{
		columns:     2,
		dataRows:    3,
		viewFromRow: 0,
		viewRows:    3,
		data: []float32{
			102, 112, 122,
			103, 113, 123,
		},
	}, vRight)
	assertMatrixEqual(t, NewMatrixFromSlices([][]float32{
		{102, 103},
		{112, 113},
		{122, 123},
	}), vRight)

	m = NewMatrixFromSlices([][]float32{
		{100, 101, 102, 103, 104, 105},
		{110, 111, 112, 113, 114, 115},
		{120, 121, 122, 123, 124, 125},
		{130, 131, 132, 133, 134, 135},
		{140, 141, 142, 143, 144, 145},
		{150, 151, 152, 153, 154, 155},
	})

	requireDeepEqual(t, Matrix{
		columns:     6,
		dataRows:    6,
		viewFromRow: 0,
		viewRows:    6,
		data: []float32{
			100, 110, 120, 130, 140, 150,
			101, 111, 121, 131, 141, 151,
			102, 112, 122, 132, 142, 152,
			103, 113, 123, 133, 143, 153,
			104, 114, 124, 134, 144, 154,
			105, 115, 125, 135, 145, 155,
		},
	}, m)

	v1 := m.View(1, 1, 4, 4)
	requireDeepEqual(t, Matrix{
		columns:     4,
		dataRows:    6,
		viewFromRow: 1,
		viewRows:    4,
		data: []float32{
			101, 111, 121, 131, 141, 151,
			102, 112, 122, 132, 142, 152,
			103, 113, 123, 133, 143, 153,
			104, 114, 124, 134, 144, 154,
		},
	}, v1)
	assertMatrixEqual(t, NewMatrixFromSlices([][]float32{
		{111, 112, 113, 114},
		{121, 122, 123, 124},
		{131, 132, 133, 134},
		{141, 142, 143, 144},
	}), v1)

	v2 := v1.View(1, 1, 2, 2)
	requireDeepEqual(t, Matrix{
		columns:     2,
		dataRows:    6,
		viewFromRow: 2,
		viewRows:    2,
		data: []float32{
			102, 112, 122, 132, 142, 152,
			103, 113, 123, 133, 143, 153,
		},
	}, v2)
	assertMatrixEqual(t, NewMatrixFromSlices([][]float32{
		{122, 123},
		{132, 133},
	}), v2)

	for i := 0; i < v1.Rows(); i++ {
		for j := 0; j < v1.Columns(); j++ {
			v1.Set(i, j, float32(800+(i*10)+j))
		}
	}
	assertMatrixEqual(t, NewMatrixFromSlices([][]float32{
		{800, 801, 802, 803},
		{810, 811, 812, 813},
		{820, 821, 822, 823},
		{830, 831, 832, 833},
	}), v1)
	assertMatrixEqual(t, NewMatrixFromSlices([][]float32{
		{811, 812},
		{821, 822},
	}), v2)

	for i := 0; i < v2.Rows(); i++ {
		for j := 0; j < v2.Columns(); j++ {
			v2.Set(i, j, float32(900+(i*10)+j))
		}
	}
	assertMatrixEqual(t, NewMatrixFromSlices([][]float32{
		{900, 901},
		{910, 911},
	}), v2)
	assertMatrixEqual(t, NewMatrixFromSlices([][]float32{
		{800, 801, 802, 803},
		{810, 900, 901, 813},
		{820, 910, 911, 823},
		{830, 831, 832, 833},
	}), v1)
	assertMatrixEqual(t, NewMatrixFromSlices([][]float32{
		{100, 101, 102, 103, 104, 105},
		{110, 800, 801, 802, 803, 115},
		{120, 810, 900, 901, 813, 125},
		{130, 820, 910, 911, 823, 135},
		{140, 830, 831, 832, 833, 145},
		{150, 151, 152, 153, 154, 155},
	}), m)
}

func TestMatrix_ViewMiddleColumns(t *testing.T) {
	m := NewMatrixFromSlices([][]float32{
		{100, 101, 102, 103},
		{110, 111, 112, 113},
		{120, 121, 122, 123},
	})
	v := m.ViewMiddleColumns(1, 2)
	assertMatrixEqual(t, NewMatrixFromSlices([][]float32{
		{101, 102},
		{111, 112},
		{121, 122},
	}), v)
}

func TestMatrix_ViewTopRows(t *testing.T) {
	m := NewMatrixFromSlices([][]float32{
		{100, 101, 102, 103},
		{110, 111, 112, 113},
		{120, 121, 122, 123},
		{130, 131, 132, 133},
	})
	v := m.ViewTopRows(2)
	assertMatrixEqual(t, NewMatrixFromSlices([][]float32{
		{100, 101, 102, 103},
		{110, 111, 112, 113},
	}), v)
}
