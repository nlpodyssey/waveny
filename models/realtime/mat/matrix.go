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
	"strings"
)

type Matrix struct {
	rows        int
	dataColumns int
	viewColumns int
	data        []float32
	scratch     []float32
}

func NewMatrix(rows, columns int) Matrix {
	size := rows * columns
	return Matrix{
		rows:        rows,
		dataColumns: columns,
		viewColumns: columns,
		data:        make([]float32, size),
		scratch:     make([]float32, size),
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
		scratch:     make([]float32, len(data)),
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
	_, _ = fmt.Fprintf(&sb, "Matrix(%d,%d)[", m.rows, m.viewColumns)
	for r := 0; r < m.rows; r++ {
		_, _ = fmt.Fprintf(&sb, "\n  %g", m.getRow(r))
	}
	sb.WriteString("\n]")
	return sb.String()
}

func (m Matrix) calcRowColumnOffset(row, column int) int {
	return row*m.dataColumns + column
}

func (m Matrix) getRow(row int) []float32 {
	from := row * m.dataColumns
	return m.data[from : from+m.viewColumns]
}
