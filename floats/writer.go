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

package floats

type Writer struct {
	slice []float32
}

func NewWriter() *Writer {
	return &Writer{}
}

func (r *Writer) Write(v float32) {
	r.slice = append(r.slice, v)
}

func (r *Writer) Floats() []float32 {
	return r.slice
}
