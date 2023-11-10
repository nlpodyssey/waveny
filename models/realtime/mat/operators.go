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

import "math"

// Product computes matrix-matrix multiplication C = A * B.
//
//go:nosplit
func Product(a, b, c Matrix) {
	aViewFromRow := a.viewFromRow
	aData := a.data
	aDataRows := a.dataRows
	cColumns := c.columns

	for i := 0; i < cColumns; i++ {
		bCol := b.getColumn(i)
		cCol := c.getColumn(i)

		for j := range cCol {
			v := float32(0)
			aOffset := aViewFromRow + j
			for _, bValue := range bCol {
				v += bValue * aData[aOffset]
				aOffset += aDataRows
			}
			cCol[j] = v
		}
	}
}

// AddProduct adds to C the result of matrix-matrix multiplication C += A * B.
//
//go:nosplit
func AddProduct(a, b, c Matrix) {
	aViewFromRow := a.viewFromRow
	aData := a.data
	aDataRows := a.dataRows
	cColumns := c.columns

	for i := 0; i < cColumns; i++ {
		bCol := b.getColumn(i)
		cCol := c.getColumn(i)

		for j := range cCol {
			v := cCol[j]
			aOffset := aViewFromRow + j
			for _, bValue := range bCol {
				v += bValue * aData[aOffset]
				aOffset += aDataRows
			}
			cCol[j] = v
		}
	}
}

// AddInPlace performs in-place element-wise addition A += B
//
//go:nosplit
func AddInPlace(a, b Matrix) {
	for i := 0; i < a.columns; i++ {
		aCol := a.getColumn(i)
		bCol := b.getColumn(i)
		_ = aCol[len(bCol)-1]
		for j, bValue := range bCol {
			aCol[j] += bValue
		}
	}
}

// AddInPlaceColumnWise adds a vector V to each column of M, in place.
// For each column c of M: M[c] += V.
//
//go:nosplit
func AddInPlaceColumnWise(m Matrix, v Vector) {
	vCol := v.getColumn(0)
	for i := 0; i < m.columns; i++ {
		mCol := m.getColumn(i)
		_ = mCol[len(vCol)-1]
		for j, v := range vCol {
			mCol[j] += v
		}
	}
}

//go:nosplit
func (m Matrix) TanhInPlace() {
	for i := 0; i < m.columns; i++ {
		mCol := m.getColumn(i)
		for j, v := range mCol {
			mCol[j] = float32(math.Tanh(float64(v)))
		}
	}
}

//go:nosplit
func (m Matrix) SigmoidInPlace() {
	for i := 0; i < m.columns; i++ {
		mCol := m.getColumn(i)
		for j, v := range mCol {
			mCol[j] = float32(1 / (1 + math.Exp(float64(-v))))
		}
	}
}

//go:nosplit
func (m Matrix) SetZero() {
	for i := 0; i < m.columns; i++ {
		mCol := m.getColumn(i)
		for j := range mCol {
			mCol[j] = 0
		}
	}
}

//go:nosplit
func Copy(destination, source Matrix) {
	for i := 0; i < destination.columns; i++ {
		copy(destination.getColumn(i), source.getColumn(i))
	}
}
