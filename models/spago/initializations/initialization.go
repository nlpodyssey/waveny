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

package initializations

import (
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
	spagorand "github.com/nlpodyssey/spago/mat/rand"
	"github.com/nlpodyssey/spago/mat/rand/uniform"
	"math"
	gorand "math/rand"
	"time"
)

func InitUniform(t mat.Tensor, from, to float64) {
	r := spagorand.NewLockedRand(uint64(time.Now().UnixNano()) + gorand.Uint64())
	u := uniform.New(from, to, r)

	shape := t.Shape()
	rows := shape[0]
	cols := shape[1]

	scalar := mat.Scalar[float32](0)

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			scalar.SetScalar(float.Interface(u.Next()), 0, 0)
			t.SetAt(scalar, i, j)
		}
	}
}

func InitKaimingUniform(t mat.Tensor, leakyReluNegativeSlope float64, receptiveFieldSize int) {
	fanIn := t.Shape()[1] * receptiveFieldSize
	gain := calculateLeakyReluGain(leakyReluNegativeSlope)
	std := gain / math.Sqrt(float64(fanIn))
	bound := math.Sqrt(3) * std
	InitUniform(t, -bound, bound)
}

func calculateLeakyReluGain(negativeSlope float64) float64 {
	return math.Sqrt(2 / (1 + math.Pow(negativeSlope, 2)))
}
