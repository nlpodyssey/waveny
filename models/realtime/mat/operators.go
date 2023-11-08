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

func Copy(destination, source Matrix) {
	for i := 0; i < destination.rows; i++ {
		copy(destination.getRow(i), source.getRow(i))
	}
}

func (m Matrix) SetZero() {
	for i := 0; i < m.rows; i++ {
		mRow := m.getRow(i)
		for j := range mRow {
			mRow[j] = 0
		}
	}
}

// Product computes matrix-matrix multiplication C = A * B.
func Product(a, b, c Matrix) {
	for i := 0; i < c.rows; i++ {
		aRow := a.getRow(i)
		cRow := c.getRow(i)
		for j := range cRow {
			v := float32(0)
			bOffset := b.viewFromColumn + j
			for k, aValue := range aRow {
				v += aValue * b.data[k*b.dataColumns+bOffset]
			}
			cRow[j] = v
		}
	}
}

// AddProduct adds to C the result of matrix-matrix multiplication C += A * B.
func AddProduct(a, b, c Matrix) {
	for i := 0; i < c.rows; i++ {
		aRow := a.getRow(i)
		cRow := c.getRow(i)
		for j := range cRow {
			v := cRow[j]
			bOffset := b.viewFromColumn + j
			for k, aValue := range aRow {
				v += aValue * b.data[k*b.dataColumns+bOffset]
			}
			cRow[j] = v
		}
	}
}

// AddInPlace performs in-place element-wise addition A += B
func AddInPlace(a, b Matrix) {
	for i := 0; i < a.rows; i++ {
		aRow := a.getRow(i)
		bRow := b.getRow(i)
		_ = aRow[len(bRow)-1]
		for j, bValue := range bRow {
			aRow[j] += bValue
		}
	}
}

// AddInPlaceColumnWise adds a vector V to each column of M, in place.
// For each column c of M: M[c] += V.
func AddInPlaceColumnWise(m Matrix, v Vector) {
	for i := 0; i < m.rows; i++ {
		vValue := v.Get(i)
		mRow := m.getRow(i)
		for j := range mRow {
			mRow[j] += vValue
		}
	}
}

func (m Matrix) TanhInPlace() {
	for i := 0; i < m.rows; i++ {
		mRow := m.getRow(i)
		for j, v := range mRow {
			mRow[j] = float32(math.Tanh(float64(v)))
		}
	}
}

func (m Matrix) SigmoidInPlace() {
	for i := 0; i < m.rows; i++ {
		mRow := m.getRow(i)
		for j, v := range mRow {
			mRow[j] = float32(1 / (1 + math.Exp(float64(-v))))
		}
	}
}
