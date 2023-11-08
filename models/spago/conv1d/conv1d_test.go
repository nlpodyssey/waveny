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

package conv1d

import (
	"fmt"
	"github.com/nlpodyssey/spago/mat"
	"math"
	"reflect"
	"testing"
)

func TestConv1D(t *testing.T) {
	testCases := []struct {
		config   Config
		weights  []mat.Tensor
		bias     mat.Tensor
		input    mat.Tensor
		expected mat.Tensor
	}{
		{
			Config{InChannels: 1, OutChannels: 1, KernelSize: 1, Dilation: 1, Bias: false},
			[]mat.Tensor{newDense(1, 1, []float32{2})},
			nil,
			newDense(1, 5, []float32{-2, -1, 0, 1, 2}),
			newDense(1, 5, []float32{-4, -2, 0, 2, 4}),
		},
		{
			Config{InChannels: 1, OutChannels: 1, KernelSize: 1, Dilation: 1, Bias: true},
			[]mat.Tensor{newDense(1, 1, []float32{2})},
			newDense(1, 1, []float32{100}),
			newDense(1, 5, []float32{-2, -1, 0, 1, 2}),
			newDense(1, 5, []float32{96, 98, 100, 102, 104}),
		},
		{
			Config{InChannels: 1, OutChannels: 1, KernelSize: 2, Dilation: 1, Bias: false},
			[]mat.Tensor{
				newDense(1, 1, []float32{2}),
				newDense(1, 1, []float32{3}),
			},
			nil,
			newDense(1, 5, []float32{-2, -1, 0, 1, 2}),
			newDense(1, 4, []float32{-7, -2, 3, 8}),
		},
		{
			Config{InChannels: 1, OutChannels: 1, KernelSize: 2, Dilation: 1, Bias: true},
			[]mat.Tensor{
				newDense(1, 1, []float32{2}),
				newDense(1, 1, []float32{3}),
			},
			newDense(1, 1, []float32{100}),
			newDense(1, 5, []float32{-2, -1, 0, 1, 2}),
			newDense(1, 4, []float32{93, 98, 103, 108}),
		},
		{
			Config{InChannels: 1, OutChannels: 1, KernelSize: 2, Dilation: 2, Bias: false},
			[]mat.Tensor{
				newDense(1, 1, []float32{2}),
				newDense(1, 1, []float32{3}),
			},
			nil,
			newDense(1, 5, []float32{-2, -1, 0, 1, 2}),
			newDense(1, 3, []float32{-4, 1, 6}),
		},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%+v", tc.config), func(t *testing.T) {
			c := New(tc.config)
			if len(tc.weights) != len(c.Weights) {
				t.Fatalf("bad weights len in test case")
			}
			for i, v := range tc.weights {
				c.Weights[i].SetData(v.Data())
			}
			if tc.config.Bias {
				c.Bias.SetData(tc.bias.Data())
			}
			y := c.Forward(tc.input)
			assertTensorInDelta(t, tc.expected, y, 0.00001)
		})
	}
}

func newDense(rows, cols int, f []float32) mat.Tensor {
	return mat.NewDense[float32](mat.WithShape(rows, cols), mat.WithBacking(f))
}

func assertTensorInDelta(t *testing.T, expected, actual mat.Tensor, delta float64, args ...any) bool {
	t.Helper()
	if expected == nil {
		t.Error("expected tensor must not be nil")
		return false
	}
	if actual == nil {
		t.Errorf("tensor values are not within delta %g:\nexpected:\n%g\nactual:\nnil\n%s",
			delta, expected.Value(), fmt.Sprint(args...))
		return false
	}
	if !reflect.DeepEqual(actual.Shape(), expected.Shape()) {
		t.Errorf("tensor values are not within delta %g because of different shapes:\nexpected shape:\n%v\nactual shape:\n%v\nexpected value:\n%g\nactual value:\n%g\n%s",
			delta, expected.Shape(), actual.Shape(), expected.Value(), actual.Value(), fmt.Sprint(args...))
		return false
	}
	expectedData := expected.Data().F64()
	actualData := actual.Data().F64()
	for i, expectedValue := range expectedData {
		actualValue := actualData[i]
		if math.Abs(expectedValue-actualValue) > delta {
			t.Errorf("tensor values are not within delta %g:\nexpected:\n%g\nactual:\n%g\n%s",
				delta, expected.Value(), actual.Value(), fmt.Sprint(args...))
			return false
		}
	}
	return true
}
