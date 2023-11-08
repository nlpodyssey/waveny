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
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
)

func ReadFile(name string) (_ *Wave, err error) {
	file, err := os.Open(name)
	if err != nil {
		return nil, fmt.Errorf("failed to open WAV file %q: %w", name, err)
	}
	defer func() {
		if e := file.Close(); e != nil && err == nil {
			err = fmt.Errorf("failed to close WAV file %q: %w", name, e)
		}
	}()
	w, err := Read(bufio.NewReader(file))
	if err != nil {
		return nil, fmt.Errorf("failed to read WAV file %q: %w", name, err)
	}
	return w, nil
}

func Read(r io.Reader) (*Wave, error) {
	var err error
	if err = expectFourCharCode(r, riffChunkID); err != nil {
		return nil, err
	}

	riffChunkSize, err := readUint32(r)
	if err != nil {
		return nil, fmt.Errorf("failed to read RIFF chunk size: %w", err)
	}
	r = io.LimitReader(r, int64(riffChunkSize))

	if err = expectFourCharCode(r, waveFormType); err != nil {
		return nil, err
	}

	format, b, err := readFormatChunkAndWaveData(r)
	if err != nil {
		return nil, err
	}

	wav := &Wave{Format: format, Data: b}
	return wav, nil
}

func readFormatChunkAndWaveData(r io.Reader) (Format, []byte, error) {
	var format Format
	formatFound := false
	for {
		chunkID, err := readFourCharCode(r)
		if err != nil {
			return Format{}, nil, err
		}
		switch chunkID {
		case fmtChunkID:
			if formatFound {
				return Format{}, nil, errors.New("duplicate format chunk encountered")
			}
			formatFound = true
			format, err = readFormatChunk(r)
			if err != nil {
				return Format{}, nil, err
			}
		case dataChunkID:
			if !formatFound {
				return Format{}, nil, errors.New("wave data is not preceded by format chunk")
			}
			b, err := readWaveData(r)
			if err != nil {
				return Format{}, nil, err
			}
			return format, b, nil
		default:
			if err = skipChunk(r, chunkID); err != nil {
				return Format{}, nil, err
			}
		}
	}
}

func readFormatChunk(r io.Reader) (Format, error) {
	chunkSize, err := readUint32(r)
	if err != nil {
		return Format{}, fmt.Errorf("failed to read format chunk size: %w", err)
	}
	r = io.LimitReader(r, int64(chunkSize))

	formatTag, err := readUint16(r)
	if err != nil {
		return Format{}, fmt.Errorf("failed to read format tag: %w", err)
	}
	if formatTag != wavePCMFormatTag {
		return Format{}, fmt.Errorf("unsupported format tag %d: only WAVE format tag %d is supported", formatTag, wavePCMFormatTag)
	}

	var format Format
	if format.Channels, err = readUint16(r); err != nil {
		return Format{}, fmt.Errorf("failed to read format's channels: %w", err)
	}
	if format.SampleRate, err = readUint32(r); err != nil {
		return Format{}, fmt.Errorf("failed to read format's' sample rate: %w", err)
	}
	if format.AvgByteRate, err = readUint32(r); err != nil {
		return Format{}, fmt.Errorf("failed to read format's' average byte rate rate: %w", err)
	}
	if format.BlockAlign, err = readUint16(r); err != nil {
		return Format{}, fmt.Errorf("failed to read format's block align: %w", err)
	}
	if format.BitsPerSample, err = readUint16(r); err != nil {
		return Format{}, fmt.Errorf("failed to read format's bits per sample: %w", err)
	}
	if err = skipUntilEOF(r); err != nil {
		return Format{}, fmt.Errorf("failed to skip remaining format chunk data: %w", err)
	}
	return format, nil
}

func readWaveData(r io.Reader) ([]byte, error) {
	chunkSize, err := readUint32(r)
	if err != nil {
		return nil, fmt.Errorf("failed to read wave data chunk size: %w", err)
	}
	b := make([]byte, chunkSize)
	_, err = io.ReadFull(r, b)
	if err != nil {
		return nil, fmt.Errorf("failed to read ave data: %w", err)
	}
	return b, nil
}

func skipChunk(r io.Reader, chunkID [4]byte) error {
	chunkSize, err := readUint32(r)
	if err != nil {
		return fmt.Errorf("failed to read %q chunk size: %w", string(chunkID[:]), err)
	}
	r = io.LimitReader(r, int64(chunkSize))
	if err = skipUntilEOF(r); err != nil {
		return fmt.Errorf("failed to skip %q chunk data: %w", string(chunkID[:]), err)
	}
	return nil
}

func expectFourCharCode(r io.Reader, expected [4]byte) error {
	code, err := readFourCharCode(r)
	if err != nil {
		return err
	}
	if code != expected {
		return fmt.Errorf("expected four-character code %q, actual %q", string(expected[:]), string(code[:]))
	}
	return nil
}

func readFourCharCode(r io.Reader) ([4]byte, error) {
	var code [4]byte
	if _, err := io.ReadFull(r, code[:]); err != nil {
		return code, fmt.Errorf("failed to read four-character code: %w", err)
	}
	return code, nil
}

func readUint16(r io.Reader) (uint16, error) {
	var arr [2]byte
	buf := arr[:]
	if _, err := io.ReadFull(r, buf); err != nil {
		return 0, fmt.Errorf("failed to read uint16 value: %w", err)
	}
	return binary.LittleEndian.Uint16(buf), nil
}

func readUint32(r io.Reader) (uint32, error) {
	var arr [4]byte
	buf := arr[:]
	if _, err := io.ReadFull(r, buf); err != nil {
		return 0, fmt.Errorf("failed to read uint32 value: %w", err)
	}
	return binary.LittleEndian.Uint32(buf), nil
}

func skipUntilEOF(r io.Reader) error {
	var arr [1]byte
	var buf = arr[:]
	for {
		_, err := r.Read(buf)
		switch err {
		case nil:
			continue
		case io.EOF:
			return nil
		default:
			return err
		}
	}
}

func Decode24BitData[T int | int32 | int64 | float32 | float64](wav *Wave) ([]T, error) {
	if wav.Format.BitsPerSample != 24 {
		return nil, fmt.Errorf("cannot convert wave data to 24-bit values: format's bits per sample is %d", wav.Format.BitsPerSample)
	}
	dataIn := wav.Data
	if len(dataIn)%3 != 0 {
		return nil, fmt.Errorf("cannot convert wave data to 24-bit values: bad data length %d", len(wav.Data))
	}
	dataOut := make([]T, len(wav.Data)/3)
	for i := range dataOut {
		j := i * 3
		dataOut[i] = T(littleEndianInt24(dataIn[j : j+3]))
	}
	return dataOut, nil
}

func littleEndianInt24(b []byte) int32 {
	_ = b[2] // bounds check hint to compiler; see golang.org/issue/14808
	// the arithmetic shifts "extend" the sign to 32-bit
	return int32(uint32(b[0])|uint32(b[1])<<8|uint32(b[2])<<16) << 8 >> 8
}
