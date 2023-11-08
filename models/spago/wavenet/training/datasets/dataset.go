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

package datasets

import (
	"fmt"
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/waveny/wave"
)

type DatasetConfig struct {
	XPath string
	YPath string
	Start int
	Stop  int
	NX    int
	NY    int
}

type Dataset struct {
	sampleRate int
	x          mat.Tensor
	y          mat.Tensor
	nx         int
	ny         int
}

func NewDataset(config DatasetConfig) (*Dataset, error) {
	x, err := wave.WavToSpagoTensor(config.XPath)
	if err != nil {
		return nil, fmt.Errorf("failed to convert X wave file to tensor: %w", err)
	}
	y, err := wave.WavToSpagoTensor(config.YPath)
	if err != nil {
		return nil, fmt.Errorf("failed to convert Y wave file to tensor: %w", err)
	}
	if !mat.SameDims(x, y) {
		return nil, fmt.Errorf("shape mismatch: X=%v, Y=%v", x.Shape(), y.Shape())
	}

	fullSize := x.Size()
	if fullSize == 0 {
		return nil, fmt.Errorf("empty training data")
	}

	start := config.Start
	if start < 0 {
		start = fullSize + start
	}
	stop := config.Stop
	if stop <= 0 {
		stop = fullSize + stop
	}
	if start < 0 || start >= fullSize || stop < 0 || stop > fullSize || stop <= start {
		return nil, fmt.Errorf("invalid [start, stop] range values [%d, %d] (data size %d)", config.Start, config.Stop, fullSize)
	}

	// TODO: require input pre-silence (default 0.4)

	x = ag.Slice(x, start, 0, stop, 1)
	y = ag.Slice(y, start, 0, stop, 1)

	// TODO: delay (default method: cubic 0.4)
	// TODO: x-scale and y-scale

	if config.NX > x.Size() {
		return nil, fmt.Errorf("invalid NX %d for input size %d", config.NX, x.Size())
	}

	if config.NY > 0 && config.NY > y.Size()-config.NX+1 {
		return nil, fmt.Errorf("invalid NY %d for input size %d and NX %d", config.NY, x.Size(), config.NX)
	}

	if ag.ReduceMax(ag.Abs(y)).Item().F32() >= 1 {
		return nil, fmt.Errorf("clipping detected in Y")
	}

	ny := config.NY
	if ny <= 0 {
		ny = x.Size() - config.NX + 1
	}
	d := &Dataset{
		sampleRate: 48000,
		x:          x,
		y:          y,
		nx:         config.NX,
		ny:         ny,
	}
	return d, nil
}

func (d *Dataset) Length() int {
	n := d.x.Size()
	singlePairs := n - d.nx + 1
	return singlePairs / d.ny
}

func (d *Dataset) YOffset() int {
	return d.nx - 1
}

func (d *Dataset) GetItem(index int) (x, y mat.Tensor) {
	if index >= d.Length() {
		panic(fmt.Sprintf("index %d out of bound (length is %d)", index, d.Length()))
	}

	i := index * d.ny
	j := i + d.YOffset()

	x = ag.Slice(d.x, i, 0, i+d.nx+d.ny-1, 1)
	y = ag.Slice(d.y, j, 0, j+d.ny, 1)
	return x, y
}
