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

package liveplay

import (
	"fmt"
	"github.com/gen2brain/malgo"
	"github.com/nlpodyssey/waveny/models/realtime/wavenet"
	"os"
	"os/signal"
	"strings"
	"unsafe"
)

type Config struct {
	ModelDataPath   string
	FramesPerBuffer int
}

func Run(config Config) (err error) {
	model, err := wavenet.LoadFromJSONModelDataFile(config.ModelDataPath)
	if err != nil {
		return err
	}

	ctx, err := malgo.InitContext(nil, malgo.ContextConfig{}, logMessage)
	if err != nil {
		return fmt.Errorf("failed to initialize malgo context: %w", err)
	}
	defer destroyContext(ctx)

	deviceConfig := makeDeviceConfig(config)

	dataProc := func(pOutputSample, pInputSamples []byte, frameCount uint32) {
		output := unsafe.Slice((*float32)(unsafe.Pointer(&pOutputSample[0])), frameCount)
		input := unsafe.Slice((*float32)(unsafe.Pointer(&pInputSamples[0])), frameCount)
		model.Process(input, output)
		model.Finalize(int(frameCount))
	}

	deviceCallbacks := malgo.DeviceCallbacks{Data: dataProc}
	device, err := malgo.InitDevice(ctx.Context, deviceConfig, deviceCallbacks)
	if err != nil {
		return fmt.Errorf("failed to initialzie malgo device: %w", err)
	}
	defer device.Uninit()

	if err = device.Start(); err != nil {
		return fmt.Errorf("failed to start malgo device: %w", err)
	}

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, os.Kill)
	sig := <-sigChan
	fmt.Printf("\nSignal: %v\n", sig)

	return nil
}

func destroyContext(ctx *malgo.AllocatedContext) {
	if err := ctx.Uninit(); err != nil {
		logMessage(fmt.Sprintf("failed to uninit malgo context: %v", err))
	}
	ctx.Free()
}

func logMessage(message string) {
	_, _ = os.Stderr.WriteString(message)
	if !strings.HasSuffix(message, "\n") {
		_, _ = os.Stderr.WriteString("\n")
	}
}

func makeDeviceConfig(config Config) malgo.DeviceConfig {
	dc := malgo.DefaultDeviceConfig(malgo.Duplex)
	dc.SampleRate = 48000
	dc.PeriodSizeInFrames = uint32(config.FramesPerBuffer)
	dc.Capture.Format = malgo.FormatF32
	dc.Capture.Channels = 1
	dc.Playback.Format = malgo.FormatF32
	dc.Playback.Channels = 1
	dc.NoClip = 1
	dc.NoPreSilencedOutputBuffer = 1
	return dc
}
