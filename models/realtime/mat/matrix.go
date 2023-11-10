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
	columns     int
	dataRows    int
	viewFromRow int
	viewRows    int
	data        []float32
}

func NewMatrix(rows, columns int) Matrix {
	return Matrix{
		columns:     columns,
		dataRows:    rows,
		viewFromRow: 0,
		viewRows:    rows,
		data:        make([]float32, rows*columns),
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
		for j, v := range rowData {
			m.Set(i, j, v)
		}
	}
	return m
}

func (m Matrix) Rows() int {
	return m.viewRows
}

func (m Matrix) Columns() int {
	return m.columns
}

func (m Matrix) Set(row, column int, value float32) {
	m.data[m.calcRowColumnOffset(row, column)] = value
}

func (m Matrix) Get(row, column int) float32 {
	return m.data[m.calcRowColumnOffset(row, column)]
}

func (m Matrix) calcRowColumnOffset(row, column int) int {
	return column*m.dataRows + m.viewFromRow + row
}

func (m Matrix) Clone() Matrix {
	if m.columns == 0 || m.viewRows == 0 {
		return Matrix{}
	}
	data := make([]float32, m.columns*m.viewRows)
	from := 0
	for i := 0; i < m.columns; i++ {
		to := from + m.viewRows
		copy(data[from:to], m.getColumn(i))
		from = to
	}
	return Matrix{
		columns:     m.columns,
		dataRows:    m.viewRows,
		viewFromRow: 0,
		viewRows:    m.viewRows,
		data:        data,
	}
}

func (m Matrix) getColumn(column int) []float32 {
	from := column*m.dataRows + m.viewFromRow
	return m.data[from : from+m.viewRows]
}

func (m Matrix) AsVector() Vector {
	return Vector{Matrix: m}
}

func (m Matrix) Resize(rows, columns int) Matrix {
	if m.columns == columns && m.viewRows == rows {
		return m
	}
	return NewMatrix(rows, columns)
}

func (m Matrix) String() string {
	sb := new(strings.Builder)
	_, _ = fmt.Fprintf(sb, "Matrix(%dx%d)[", m.viewRows, m.columns)

	for i := 0; i < m.viewRows; i++ {
		sb.WriteString("\n  [")
		for j := 0; j < m.columns; j++ {
			_, _ = fmt.Fprintf(sb, " %g", m.Get(i, j))
		}
		sb.WriteString(" ]")
	}

	sb.WriteString("\n]")
	return sb.String()
}
