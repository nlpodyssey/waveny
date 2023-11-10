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
		rows:        3,
		dataColumns: 4,
		viewColumns: 4,
		data: []float32{
			100, 101, 102, 103,
			110, 111, 112, 113,
			120, 121, 122, 123,
		},
		qData:     make([]int8, 12),
		quantized: false,
		qScale:    0,
	}, m)

	vIdentity := m.View(0, 0, 3, 4)
	requireDeepEqual(t, m, vIdentity)

	vTop := m.View(0, 0, 2, 4)
	requireDeepEqual(t, Matrix{
		rows:        2,
		dataColumns: 4,
		viewColumns: 4,
		data: []float32{
			100, 101, 102, 103,
			110, 111, 112, 113,
		},
		qData:     make([]int8, 8),
		quantized: false,
		qScale:    0,
	}, vTop)
	assertMatrixEqual(t, NewMatrixFromSlices([][]float32{
		{100, 101, 102, 103},
		{110, 111, 112, 113},
	}), vTop)

	vBottom := m.View(1, 0, 2, 4)
	requireDeepEqual(t, Matrix{
		rows:        2,
		dataColumns: 4,
		viewColumns: 4,
		data: []float32{
			110, 111, 112, 113,
			120, 121, 122, 123,
		},
		qData:     make([]int8, 8),
		quantized: false,
		qScale:    0,
	}, vBottom)
	assertMatrixEqual(t, NewMatrixFromSlices([][]float32{
		{110, 111, 112, 113},
		{120, 121, 122, 123},
	}), vBottom)

	vLeft := m.View(0, 0, 3, 2)
	requireDeepEqual(t, Matrix{
		rows:        3,
		dataColumns: 4,
		viewColumns: 2,
		data: []float32{
			100, 101, 102, 103,
			110, 111, 112, 113,
			120, 121,
		},
		qData:     make([]int8, 10),
		quantized: false,
		qScale:    0,
	}, vLeft)
	assertMatrixEqual(t, NewMatrixFromSlices([][]float32{
		{100, 101},
		{110, 111},
		{120, 121},
	}), vLeft)

	vRight := m.View(0, 2, 3, 2)
	requireDeepEqual(t, Matrix{
		rows:        3,
		dataColumns: 4,
		viewColumns: 2,
		data: []float32{
			102, 103,
			110, 111, 112, 113,
			120, 121, 122, 123,
		},
		qData:     make([]int8, 10),
		quantized: false,
		qScale:    0,
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
		rows:        6,
		dataColumns: 6,
		viewColumns: 6,
		data: []float32{
			100, 101, 102, 103, 104, 105,
			110, 111, 112, 113, 114, 115,
			120, 121, 122, 123, 124, 125,
			130, 131, 132, 133, 134, 135,
			140, 141, 142, 143, 144, 145,
			150, 151, 152, 153, 154, 155,
		},
		qData:     make([]int8, 6*6),
		quantized: false,
		qScale:    0,
	}, m)

	v1 := m.View(1, 1, 4, 4)
	requireDeepEqual(t, Matrix{
		rows:        4,
		dataColumns: 6,
		viewColumns: 4,
		data: []float32{
			111, 112, 113, 114, 115,
			120, 121, 122, 123, 124, 125,
			130, 131, 132, 133, 134, 135,
			140, 141, 142, 143, 144,
		},
		qData:     make([]int8, 22),
		quantized: false,
		qScale:    0,
	}, v1)
	assertMatrixEqual(t, NewMatrixFromSlices([][]float32{
		{111, 112, 113, 114},
		{121, 122, 123, 124},
		{131, 132, 133, 134},
		{141, 142, 143, 144},
	}), v1)

	v2 := v1.View(1, 1, 2, 2)
	requireDeepEqual(t, Matrix{
		rows:        2,
		dataColumns: 6,
		viewColumns: 2,
		data: []float32{
			122, 123, 124, 125,
			130, 131, 132, 133,
		},
		qData:     make([]int8, 8),
		quantized: false,
		qScale:    0,
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

	vLeft := m.ViewMiddleColumns(0, 2)
	requireDeepEqual(t, Matrix{
		rows:        3,
		dataColumns: 4,
		viewColumns: 2,
		data: []float32{
			100, 101, 102, 103,
			110, 111, 112, 113,
			120, 121,
		},
		qData:     make([]int8, 10),
		quantized: false,
		qScale:    0,
	}, vLeft)
	assertMatrixEqual(t, NewMatrixFromSlices([][]float32{
		{100, 101},
		{110, 111},
		{120, 121},
	}), vLeft)

	vRight := m.ViewMiddleColumns(2, 2)
	requireDeepEqual(t, Matrix{
		rows:        3,
		dataColumns: 4,
		viewColumns: 2,
		data: []float32{
			102, 103,
			110, 111, 112, 113,
			120, 121, 122, 123,
		},
		qData:     make([]int8, 10),
		quantized: false,
		qScale:    0,
	}, vRight)
	assertMatrixEqual(t, NewMatrixFromSlices([][]float32{
		{102, 103},
		{112, 113},
		{122, 123},
	}), vRight)

	vMiddle := m.ViewMiddleColumns(1, 2)
	requireDeepEqual(t, Matrix{
		rows:        3,
		dataColumns: 4,
		viewColumns: 2,
		data: []float32{
			101, 102, 103,
			110, 111, 112, 113,
			120, 121, 122,
		},
		qData:     make([]int8, 10),
		quantized: false,
		qScale:    0,
	}, vMiddle)
	assertMatrixEqual(t, NewMatrixFromSlices([][]float32{
		{101, 102},
		{111, 112},
		{121, 122},
	}), vMiddle)
}
