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
	"fmt"
	"math"
	"strings"
)

type Matrix struct {
	rows        int
	dataColumns int
	viewColumns int
	data        []float32
	qData       []int8
	quantized   bool
	qScale      float32
}

func NewMatrix(rows, columns int) Matrix {
	size := rows * columns
	return Matrix{
		rows:        rows,
		dataColumns: columns,
		viewColumns: columns,
		data:        make([]float32, size),
		qData:       make([]int8, size),
		quantized:   false,
		qScale:      0,
	}
}

func NewMatrixFromSlices(data [][]float32) Matrix {
	rows := len(data)
	if rows == 0 {
		return Matrix{}
	}
	columns := len(data[0])
	m := NewMatrix(rows, columns)
	for i, rowData := range data {
		copy(m.data[i*columns:i*columns+columns], rowData)
	}
	return m
}

func (m Matrix) Rows() int {
	return m.rows
}

func (m Matrix) Columns() int {
	return m.viewColumns
}

func (m Matrix) Set(row, column int, value float32) {
	m.data[m.calcRowColumnOffset(row, column)] = value
}

func (m Matrix) Get(row, column int) float32 {
	return m.data[m.calcRowColumnOffset(row, column)]
}

func (m Matrix) Clone() Matrix {
	if m.rows == 0 || m.viewColumns == 0 {
		return Matrix{}
	}
	data := make([]float32, m.rows*m.viewColumns)
	for i := 0; i < m.rows; i++ {
		from := i * m.viewColumns
		copy(data[from:from+m.viewColumns], m.getRow(i))
	}
	return Matrix{
		rows:        m.rows,
		dataColumns: m.viewColumns,
		viewColumns: m.viewColumns,
		data:        data,
		qData:       make([]int8, len(data)),
		quantized:   false,
		qScale:      0,
	}
}

func (m Matrix) AsVector() Vector {
	return Vector{Matrix: m}
}

func (m Matrix) Resize(rows, columns int) Matrix {
	if m.rows == rows && m.viewColumns == columns {
		return m
	}
	return NewMatrix(rows, columns)
}

func (m Matrix) String() string {
	sb := strings.Builder{}
	if m.quantized {
		_, _ = fmt.Fprintf(&sb, "Matrix(%d,%d)Q(%g)[", m.rows, m.viewColumns, m.qScale)
		for r := 0; r < m.rows; r++ {
			_, _ = fmt.Fprintf(&sb, "\n  %d", m.getQRow(r))
		}
		sb.WriteString("\n]")
	} else {
		_, _ = fmt.Fprintf(&sb, "Matrix(%d,%d)[", m.rows, m.viewColumns)
		for r := 0; r < m.rows; r++ {
			_, _ = fmt.Fprintf(&sb, "\n  %g", m.getRow(r))
		}
		sb.WriteString("\n]")
	}
	return sb.String()
}

func (m Matrix) calcRowColumnOffset(row, column int) int {
	return row*m.dataColumns + column
}

func (m Matrix) getRow(row int) []float32 {
	from := row * m.dataColumns
	return m.data[from : from+m.viewColumns]
}

func (m Matrix) getQRow(row int) []int8 {
	from := row * m.dataColumns
	return m.qData[from : from+m.viewColumns]
}

func (m Matrix) Quantized() Matrix {
	if m.quantized {
		return m
	}

	alpha := m.findAbsMax()
	if alpha == 0 {
		alpha = 0.001
	}
	s := 127 / alpha

	from := 0
	for i := 0; i < m.rows; i++ {
		to := from + m.viewColumns
		row := m.data[from:to]
		qRow := m.qData[from:to]
		from += m.dataColumns
		_ = qRow[len(row)-1]

		for j, v := range row {
			qRow[j] = int8(clipF32(roundF32(s*v), -127, 127))
		}
	}

	return Matrix{
		rows:        m.rows,
		dataColumns: m.dataColumns,
		viewColumns: m.viewColumns,
		data:        m.data,
		qData:       m.qData,
		quantized:   true,
		qScale:      s,
	}
}

func (m Matrix) asQuantized(s float32) Matrix {
	return Matrix{
		rows:        m.rows,
		dataColumns: m.dataColumns,
		viewColumns: m.viewColumns,
		data:        m.data,
		qData:       m.qData,
		quantized:   true,
		qScale:      s,
	}
}

func (m Matrix) DeQuantized() Matrix {
	s := m.qScale

	from := 0
	for i := 0; i < m.rows; i++ {
		to := from + m.viewColumns
		row := m.data[from:to]
		qRow := m.qData[from:to]
		from += m.dataColumns
		_ = row[len(qRow)-1]

		for j, v := range qRow {
			row[j] = float32(v) / s
		}
	}

	return Matrix{
		rows:        m.rows,
		dataColumns: m.dataColumns,
		viewColumns: m.viewColumns,
		data:        m.data,
		qData:       m.qData,
		quantized:   false,
		qScale:      0,
	}
}

func (m Matrix) findAbsMax() float32 {
	absMax := absF32(m.Get(0, 0))
	for r := 0; r < m.rows; r++ {
		for _, v := range m.getRow(r) {
			absMax = max(absMax, absF32(v))
		}
	}
	return absMax
}

func absF32(v float32) float32 {
	return float32(math.Abs(float64(v)))
}

func roundF32(v float32) float32 {
	return float32(math.Round(float64(v)))
}

func clipF32(v, lo, hi float32) float32 {
	return max(lo, min(hi, v))
}

func clipI32(v, lo, hi int32) int32 {
	return max(lo, min(hi, v))
}
