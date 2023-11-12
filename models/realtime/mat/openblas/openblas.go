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

package openblas

/*
#cgo pkg-config: openblas
#include <cblas.h>
*/
import "C"
import "unsafe"

type Order = C.enum_CBLAS_ORDER

const (
	RowMajor Order = C.CblasRowMajor
	ColMajor Order = C.CblasColMajor
)

type Transpose = C.enum_CBLAS_TRANSPOSE

const (
	NoTrans     Transpose = C.CblasNoTrans
	Trans       Transpose = C.CblasTrans
	ConjTrans   Transpose = C.CblasConjTrans
	ConjNoTrans Transpose = C.CblasConjNoTrans
)

type (
	blasint = C.blasint
	float   = C.float
)

func Sgemm(
	order Order,
	transA Transpose,
	transB Transpose,
	m int,
	n int,
	k int,
	alpha float32,
	a []float32,
	lda int,
	b []float32,
	ldb int,
	beta float32,
	c []float32,
	ldc int,
) {
	C.cblas_sgemm(
		order,
		transA,
		transB,
		blasint(m),
		blasint(n),
		blasint(k),
		float(alpha),
		(*float)(unsafe.Pointer(&a[0])),
		blasint(lda),
		(*float)(unsafe.Pointer(&b[0])),
		blasint(ldb),
		float(beta),
		(*float)(unsafe.Pointer(&c[0])),
		blasint(ldc),
	)
}
