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
)

//go:nosplit
func Copy(destination, source Matrix) {
	for i := 0; i < destination.rows; i++ {
		copy(destination.getRow(i), source.getRow(i))
	}
}

//go:nosplit
func (m Matrix) SetZero() {
	for i := 0; i < m.rows; i++ {
		mRow := m.getRow(i)
		for j := range mRow {
			mRow[j] = 0
		}
	}
}

// Product computes matrix-matrix multiplication C = A * B.
//
//go:nosplit
func Product(a, b, c Matrix) {
	aData, _ := a.makeContiguousData()
	bTData, _ := b.makeContiguousTransposedData()

	cData := c.data
	cIsScratch := c.dataColumns != c.viewColumns
	if cIsScratch {
		cData = c.scratch
	}

	acRows := c.rows
	aColumns := a.viewColumns
	cColumns := c.viewColumns
	bRows := b.rows

	for i := 0; i < acRows; i++ {
		aRow := aData[i*aColumns : i*aColumns+aColumns]
		cRow := cData[i*cColumns : i*cColumns+cColumns]
		for j := range cRow {
			v := float32(0)

			bColumn := bTData[j*bRows : j*bRows+bRows]
			_ = bColumn[len(aRow)-1]
			for k, aValue := range aRow {
				v += aValue * bColumn[k]
			}

			cRow[j] = v
		}
	}

	if cIsScratch {
		c.copyFromContiguousData(cData)
	}
}

// AddProduct adds to C the result of matrix-matrix multiplication C += A * B.
//
//go:nosplit
func AddProduct(a, b, c Matrix) {
	aData, _ := a.makeContiguousData()
	bTData, _ := b.makeContiguousTransposedData()
	cData, cIsScratch := c.makeContiguousData()

	acRows := c.rows
	aColumns := a.viewColumns
	cColumns := c.viewColumns
	bRows := b.rows

	for i := 0; i < acRows; i++ {
		aRow := aData[i*aColumns : i*aColumns+aColumns]
		cRow := cData[i*cColumns : i*cColumns+cColumns]
		for j := range cRow {
			v := cRow[j]

			bColumn := bTData[j*bRows : j*bRows+bRows]
			_ = bColumn[len(aRow)-1]
			for k, aValue := range aRow {
				v += aValue * bColumn[k]
			}

			cRow[j] = v
		}
	}

	if cIsScratch {
		c.copyFromContiguousData(cData)
	}
}

//go:nosplit
func (m Matrix) makeContiguousData() ([]float32, bool) {
	if m.dataColumns == m.viewColumns {
		return m.data, false
	}

	rows := m.rows
	viewColumns := m.viewColumns
	dataColumns := m.dataColumns

	data := m.data
	scratch := m.scratch

	dataFrom := 0
	scratchFrom := 0
	for i := 0; i < rows; i++ {
		dataTo := dataFrom + viewColumns
		scratchTo := scratchFrom + viewColumns
		copy(scratch[scratchFrom:scratchTo], data[dataFrom:dataTo])
		dataFrom += dataColumns
		scratchFrom += viewColumns
	}
	return scratch, true
}

//go:nosplit
func (m Matrix) makeContiguousTransposedData() ([]float32, bool) {
	if m.rows == 1 || m.viewColumns == 1 {
		return m.makeContiguousData()
	}

	rows := m.rows
	viewColumns := m.viewColumns
	dataColumns := m.dataColumns

	data := m.data
	scratch := m.scratch

	dataFrom := 0
	for i := 0; i < rows; i++ {
		dataTo := dataFrom + viewColumns
		dataRow := data[dataFrom:dataTo]

		scratchOffset := i
		for _, v := range dataRow {
			scratch[scratchOffset] = v
			scratchOffset += rows
		}
		dataFrom += dataColumns
	}

	return scratch, true
}

//go:nosplit
func (m Matrix) copyFromContiguousData(scratch []float32) {
	rows := m.rows
	viewColumns := m.viewColumns
	dataColumns := m.dataColumns

	data := m.data

	dataFrom := 0
	scratchFrom := 0
	for i := 0; i < rows; i++ {
		dataTo := dataFrom + viewColumns
		scratchTo := scratchFrom + viewColumns
		copy(data[dataFrom:dataTo], scratch[scratchFrom:scratchTo])
		dataFrom += dataColumns
		scratchFrom += viewColumns
	}
}

// AddInPlace performs in-place element-wise addition A += B
//
//go:nosplit
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
//
//go:nosplit
func AddInPlaceColumnWise(m Matrix, v Vector) {
	for i := 0; i < m.rows; i++ {
		vValue := v.Get(i)
		mRow := m.getRow(i)
		for j := range mRow {
			mRow[j] += vValue
		}
	}
}

//go:nosplit
func (m Matrix) TanhInPlace() {
	for i := 0; i < m.rows; i++ {
		mRow := m.getRow(i)
		for j, v := range mRow {
			mRow[j] = float32(math.Tanh(float64(v)))
		}
	}
}

//go:nosplit
func (m Matrix) SigmoidInPlace() {
	for i := 0; i < m.rows; i++ {
		mRow := m.getRow(i)
		for j, v := range mRow {
			mRow[j] = float32(1 / (1 + math.Exp(float64(-v))))
		}
	}
}
