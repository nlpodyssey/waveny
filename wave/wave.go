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

package wave

import (
	"fmt"
	"github.com/nlpodyssey/spago/mat"
	"math"
)

type Wave struct {
	Format Format
	Data   []byte
}

type Format struct {
	Channels      uint16
	SampleRate    uint32
	AvgByteRate   uint32
	BlockAlign    uint16
	BitsPerSample uint16
}

var (
	wavePCMFormatTag uint16 = 1
	riffChunkID             = [4]byte{'R', 'I', 'F', 'F'}
	waveFormType            = [4]byte{'W', 'A', 'V', 'E'}
	fmtChunkID              = [4]byte{'f', 'm', 't', ' '}
	dataChunkID             = [4]byte{'d', 'a', 't', 'a'}
)

func ComputePCMBlockAlign(channels, bitsPerSample int) uint16 {
	return uint16(math.Ceil(float64(channels) * float64(bitsPerSample) / 8))
}

func ComputePCMBAvgByteRate(channels, bitsPerSample, sampleRate int) uint32 {
	return uint32(math.Ceil(float64(channels) * float64(sampleRate) * float64(bitsPerSample) / 8))
}

const scaling24bit = 2 << 22 // 2 ** (24 - 1)

func WavToFloats(filename string) ([]float32, error) {
	wav, err := ReadFile(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to read WAV file: %w", err)
	}
	if wav.Format.Channels != 1 {
		return nil, fmt.Errorf("only 1 channel (mono) is supported, actual: %d", wav.Format.Channels)
	}
	if wav.Format.SampleRate != 48_000 {
		return nil, fmt.Errorf("only sample rate 48000 is supported, actual: %d", wav.Format.SampleRate)
	}
	if wav.Format.BitsPerSample != 24 {
		return nil, fmt.Errorf("only 24-bit samples are supported, actual: %d", wav.Format.BitsPerSample)
	}

	data, err := Decode24BitData[float32](wav)
	if err != nil {
		return nil, err
	}

	for i := range data {
		data[i] /= scaling24bit
	}
	return data, nil
}

func WavToSpagoTensor(filename string) (mat.Tensor, error) {
	data, err := WavToFloats(filename)
	if err != nil {
		return nil, err
	}
	m := mat.NewDense[float32](mat.WithBacking(data))
	return m, nil
}

func FloatsToWav(data []float32, filename string) error {
	for i, v := range data {
		data[i] = clip(v, -1, 1) * scaling24bit
	}
	wav := Wave{
		Format: Format{
			Channels:      1,
			SampleRate:    48_000,
			AvgByteRate:   ComputePCMBAvgByteRate(1, 24, 48_000),
			BlockAlign:    ComputePCMBlockAlign(1, 24),
			BitsPerSample: 24,
		},
		Data: Encode24BitData(data),
	}
	if err := WriteFile(&wav, filename); err != nil {
		return fmt.Errorf("failed to write WAV file: %w", err)
	}
	return nil
}

func SpagoTensorToWav(tensor mat.Tensor, filename string) error {
	return FloatsToWav(tensor.Data().F32(), filename)
}

func clip(v, vMin, vMax float32) float32 {
	return max(vMin, min(v, vMax))
}
