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
	"testing"
)

func TestProduct(t *testing.T) {
	testCases := []struct {
		name     string
		a        Matrix
		b        Matrix
		expected Matrix
	}{
		{
			"1x1",
			NewVectorFromSlice([]float32{2}).Matrix,
			NewVectorFromSlice([]float32{3}).Matrix,
			NewVectorFromSlice([]float32{6}).Matrix,
		},
		{
			"1x2 * 2x1",
			NewMatrixFromSlices([][]float32{{2, 3}}),
			NewMatrixFromSlices([][]float32{{4}, {5}}),
			NewMatrixFromSlices([][]float32{{23}}),
		},
		{
			"2x3 * 3x4",
			NewMatrixFromSlices([][]float32{
				{10, 20, 30},
				{40, 50, 60}}),
			NewMatrixFromSlices([][]float32{
				{1, 2, 3, 4},
				{5, 6, 7, 8},
				{9, 10, 11, 12}}),
			NewMatrixFromSlices([][]float32{
				{380, 440, 500, 560},
				{830, 980, 1130, 1280}}),
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			actual := NewMatrix(tc.a.Rows(), tc.b.Columns())
			Product(tc.a, tc.b, actual)
			assertMatrixEqual(t, tc.expected, actual)
		})
	}

	t.Run("views", func(t *testing.T) {
		a := NewMatrixFromSlices([][]float32{
			{9, 9, 9, 9, 9},
			{9, 10, 20, 30, 9},
			{9, 40, 50, 60, 9},
			{9, 9, 9, 9, 9},
		}).View(1, 1, 2, 3)
		b := NewMatrixFromSlices([][]float32{
			{8, 8, 8, 8, 8, 8},
			{8, 1, 2, 3, 4, 8},
			{8, 5, 6, 7, 8, 8},
			{8, 9, 10, 11, 12, 8},
			{8, 8, 8, 8, 8, 8},
		}).View(1, 1, 3, 4)
		c := NewMatrixFromSlices([][]float32{
			{9, 9, 9, 9, 9, 9},
			{9, 9, 9, 9, 9, 9},
			{9, 9, 9, 9, 9, 9},
			{9, 9, 9, 9, 9, 9}})
		vc := c.View(1, 1, 2, 4)
		Product(a, b, vc)
		expected := NewMatrixFromSlices([][]float32{
			{9, 9, 9, 9, 9, 9},
			{9, 380, 440, 500, 560, 9},
			{9, 830, 980, 1130, 1280, 9},
			{9, 9, 9, 9, 9, 9}})
		assertMatrixEqual(t, expected, c)
	})
}

func TestAddProduct(t *testing.T) {
	testCases := []struct {
		name     string
		a        Matrix
		b        Matrix
		c        Matrix
		expected Matrix
	}{
		{
			"1x1",
			NewVectorFromSlice([]float32{2}).Matrix,
			NewVectorFromSlice([]float32{3}).Matrix,
			NewVectorFromSlice([]float32{10}).Matrix,
			NewVectorFromSlice([]float32{16}).Matrix,
		},
		{
			"1x2 * 2x1",
			NewMatrixFromSlices([][]float32{{2, 3}}),
			NewMatrixFromSlices([][]float32{{4}, {5}}),
			NewMatrixFromSlices([][]float32{{100}}),
			NewMatrixFromSlices([][]float32{{123}}),
		},
		{
			"2x3 * 3x4",
			NewMatrixFromSlices([][]float32{
				{10, 20, 30},
				{40, 50, 60}}),
			NewMatrixFromSlices([][]float32{
				{1, 2, 3, 4},
				{5, 6, 7, 8},
				{9, 10, 11, 12}}),
			NewMatrixFromSlices([][]float32{
				{.1, .2, .3, .4},
				{.5, .6, .7, .8}}),
			NewMatrixFromSlices([][]float32{
				{380.1, 440.2, 500.3, 560.4},
				{830.5, 980.6, 1130.7, 1280.8}}),
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			actual := tc.c.Clone()
			AddProduct(tc.a, tc.b, actual)
			assertMatrixEqual(t, tc.expected, actual)
		})
	}

	t.Run("views", func(t *testing.T) {
		a := NewMatrixFromSlices([][]float32{
			{9, 9, 9, 9, 9},
			{9, 10, 20, 30, 9},
			{9, 40, 50, 60, 9},
			{9, 9, 9, 9, 9},
		}).View(1, 1, 2, 3)
		b := NewMatrixFromSlices([][]float32{
			{8, 8, 8, 8, 8, 8},
			{8, 1, 2, 3, 4, 8},
			{8, 5, 6, 7, 8, 8},
			{8, 9, 10, 11, 12, 8},
			{8, 8, 8, 8, 8, 8},
		}).View(1, 1, 3, 4)
		c := NewMatrixFromSlices([][]float32{
			{9, 9, 9, 9, 9, 9},
			{9, .1, .2, .3, .4, 9},
			{9, .5, .6, .7, .8, 9},
			{9, 9, 9, 9, 9, 9}})
		vc := c.View(1, 1, 2, 4)
		AddProduct(a, b, vc)
		expected := NewMatrixFromSlices([][]float32{
			{9, 9, 9, 9, 9, 9},
			{9, 380.1, 440.2, 500.3, 560.4, 9},
			{9, 830.5, 980.6, 1130.7, 1280.8, 9},
			{9, 9, 9, 9, 9, 9}})
		assertMatrixEqual(t, expected, c)
	})
}

func TestAddInPlace(t *testing.T) {
	testCases := []struct {
		name     string
		a        Matrix
		b        Matrix
		expected Matrix
	}{
		{
			"matrices",
			NewMatrixFromSlices([][]float32{
				{1, 2, 3},
				{4, 5, 6}}),
			NewMatrixFromSlices([][]float32{
				{.1, .2, .3},
				{.4, .5, .6}}),
			NewMatrixFromSlices([][]float32{
				{1.1, 2.2, 3.3},
				{4.4, 5.5, 6.6}}),
		},
		{
			"views",
			NewMatrixFromSlices([][]float32{
				{9, 9, 9, 9, 9},
				{9, 1, 2, 3, 9},
				{9, 4, 5, 6, 9},
				{9, 9, 9, 9, 9},
			}).View(1, 1, 2, 3),
			NewMatrixFromSlices([][]float32{
				{8, 8, 8, 8, 8},
				{8, .1, .2, .3, 8},
				{8, .4, .5, .6, 8},
				{8, 8, 8, 8, 8},
			}).View(1, 1, 2, 3),
			NewMatrixFromSlices([][]float32{
				{1.1, 2.2, 3.3},
				{4.4, 5.5, 6.6}}),
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			actual := tc.a.Clone()
			AddInPlace(actual, tc.b)
			assertMatrixEqual(t, tc.expected, actual)
		})
	}
}

func TestAddInPlaceColumnWise(t *testing.T) {
	testCases := []struct {
		name     string
		m        Matrix
		v        Vector
		expected Matrix
	}{
		{
			"matrices",
			NewMatrixFromSlices([][]float32{
				{1, 2},
				{3, 4},
				{5, 6}}),
			NewVectorFromSlice([]float32{70, 80, 90}),
			NewMatrixFromSlices([][]float32{
				{71, 72},
				{83, 84},
				{95, 96}}),
		},
		{
			"views",
			NewMatrixFromSlices([][]float32{
				{9, 9, 9, 9},
				{9, 1, 2, 9},
				{9, 3, 4, 9},
				{9, 5, 6, 9},
				{9, 9, 9, 9},
			}).View(1, 1, 3, 2),
			NewMatrixFromSlices([][]float32{
				{8, 8, 8},
				{8, 70, 8},
				{8, 80, 8},
				{8, 90, 8},
				{8, 8, 8},
			}).View(1, 1, 3, 1).AsVector(),
			NewMatrixFromSlices([][]float32{
				{71, 72},
				{83, 84},
				{95, 96}}),
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			actual := tc.m.Clone()
			AddInPlaceColumnWise(actual, tc.v)
			assertMatrixEqual(t, tc.expected, actual)
		})
	}
}
