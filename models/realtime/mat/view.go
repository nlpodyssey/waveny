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

func (m Matrix) View(fromRow, fromColumn, numRows, numColumns int) Matrix {
	start := fromRow*m.dataColumns + fromColumn
	return Matrix{
		rows:        numRows,
		dataColumns: m.dataColumns,
		viewColumns: numColumns,
		data:        m.data[start : start+(numRows*m.dataColumns)-(m.dataColumns-numColumns)],
	}
}

func (m Matrix) ViewMiddleColumns(fromColumn, numColumns int) Matrix {
	return m.View(0, fromColumn, m.rows, numColumns)
}

func (m Matrix) ViewTopRows(n int) Matrix {
	return m.View(0, 0, n, m.viewColumns)
}
