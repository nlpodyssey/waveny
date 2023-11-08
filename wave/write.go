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
	"fmt"
	"io"
	"os"
)

func WriteFile(wav *Wave, name string) (err error) {
	file, err := os.Create(name)
	if err != nil {
		return fmt.Errorf("failed to create WAV file %q: %w", name, err)
	}
	defer func() {
		if e := file.Close(); e != nil && err == nil {
			err = fmt.Errorf("failed to close WAV file %q: %w", name, e)
		}
	}()
	bw := bufio.NewWriter(file)
	err = Write(wav, bw)
	if err != nil {
		return fmt.Errorf("failed to write WAV file %q: %w", name, err)
	}
	if err = bw.Flush(); err != nil {
		return fmt.Errorf("failed to flush buffer for WAV file %q: %w", name, err)
	}
	return nil
}

func Write(wav *Wave, w io.Writer) error {
	if err := writeFourCharCode(w, riffChunkID); err != nil {
		return err
	}
	if err := writeUint32(w, computeRIFFChunkSize(wav)); err != nil {
		return err
	}
	if err := writeFourCharCode(w, waveFormType); err != nil {
		return err
	}
	if err := writeFormatChunk(w, wav); err != nil {
		return err
	}
	return writeWaveData(w, wav)
}

func writeFormatChunk(w io.Writer, wav *Wave) error {
	if err := writeFourCharCode(w, fmtChunkID); err != nil {
		return err
	}
	if err := writeUint32(w, formatChunkSize); err != nil {
		return err
	}
	if err := writeUint16(w, wavePCMFormatTag); err != nil {
		return err
	}
	format := wav.Format
	if err := writeUint16(w, format.Channels); err != nil {
		return err
	}
	if err := writeUint32(w, format.SampleRate); err != nil {
		return err
	}
	if err := writeUint32(w, format.AvgByteRate); err != nil {
		return err
	}
	if err := writeUint16(w, format.BlockAlign); err != nil {
		return err
	}
	if err := writeUint16(w, format.BitsPerSample); err != nil {
		return err
	}
	return nil
}

func writeWaveData(w io.Writer, wav *Wave) error {
	if err := writeFourCharCode(w, dataChunkID); err != nil {
		return err
	}
	if err := writeUint32(w, uint32(len(wav.Data))); err != nil {
		return err
	}
	if _, err := w.Write(wav.Data); err != nil {
		return fmt.Errorf("failed to write wave data: %w", err)
	}
	return nil
}

const formatChunkSize = 2 + // format tag
	2 + // channels
	4 + // sample rate
	4 + // average byte rate
	2 + // block align
	2 // bits per sample

func computeRIFFChunkSize(wav *Wave) uint32 {
	return 4 + // "WAVE"
		4 + // "fmt "
		4 + // fmt size
		formatChunkSize +
		4 + // "data"
		4 + // data size
		uint32(len(wav.Data))
}

func writeFourCharCode(w io.Writer, value [4]byte) error {
	_, err := w.Write(value[:])
	if err != nil {
		return fmt.Errorf("failed to write four-character code %q: %w", string(value[:]), err)
	}
	return nil
}

func Encode24BitData[T int | int32 | int64 | float32 | float64](dataIn []T) []byte {
	dataOut := make([]byte, len(dataIn)*3)
	for i, v := range dataIn {
		j := i * 3
		putLittleEndianInt24(dataOut[j:j+3], int32(v))
	}
	return dataOut
}

func putLittleEndianInt24(b []byte, v int32) {
	u := uint32(v)
	_ = b[2] // early bounds check to guarantee safety of writes below
	b[0] = byte(u)
	b[1] = byte(u >> 8)
	b[2] = byte(u >> 16)
}

func writeUint16(w io.Writer, v uint16) error {
	var arr [2]byte
	buf := arr[:]
	binary.LittleEndian.PutUint16(buf, v)
	_, err := w.Write(buf)
	if err != nil {
		return fmt.Errorf("failed to write uint16 value: %w", err)
	}
	return nil
}

func writeUint32(w io.Writer, v uint32) error {
	var arr [4]byte
	buf := arr[:]
	binary.LittleEndian.PutUint32(buf, v)
	_, err := w.Write(buf)
	if err != nil {
		return fmt.Errorf("failed to write uint32 value: %w", err)
	}
	return nil
}
