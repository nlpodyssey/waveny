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

package activations

import (
	"fmt"
	"github.com/nlpodyssey/waveny/models/realtime/mat"
)

type Activation interface {
	Apply(matrix mat.Matrix)
}

func New(name string) Activation {
	switch name {
	case "Sigmoid":
		return NewSigmoid()
	case "Tanh":
		return NewTanh()
	default:
		panic(fmt.Sprintf("unknown or unsupported activation %q", name))
	}
}
