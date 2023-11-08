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
	"github.com/gordonklaus/portaudio"
	"github.com/nlpodyssey/waveny/models/realtime/wavenet"
	"os"
	"os/signal"
)

const (
	numInputChannels  = 1
	numOutputChannels = 1
	sampleRate        = 48000
	framesPerBuffer   = 256
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

	if err = portaudio.Initialize(); err != nil {
		return fmt.Errorf("failed to initialize PortAudio: %w", err)
	}
	defer func() {
		if e := portaudio.Terminate(); e != nil && err == nil {
			err = fmt.Errorf("failed to terminate PortAudio: %w", err)
		}
	}()

	process := func(input, output []float32) {
		model.Process(input, output)
		model.Finalize(len(input))
	}

	stream, err := portaudio.OpenDefaultStream(
		numInputChannels,
		numOutputChannels,
		sampleRate,
		framesPerBuffer,
		process,
	)
	if err != nil {
		return fmt.Errorf("failed to open default PortAudio stream: %w", err)
	}
	defer func() {
		if e := stream.Close(); e != nil && err == nil {
			err = fmt.Errorf("failed to close PortAudio stream: %w", err)
		}
	}()

	if err = stream.Start(); err != nil {
		return fmt.Errorf("failed to start PortAudio stream: %w", err)
	}
	fmt.Println("PortAudio stream started.\nCtrl+C / SIGINT / SIGKILL to quit.")

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, os.Kill)
	sig := <-sigChan
	fmt.Printf("\nSignal: %v\n", sig)

	return nil
}
