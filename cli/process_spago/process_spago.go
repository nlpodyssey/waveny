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

package process_spago

import (
	"errors"
	"flag"
	"github.com/nlpodyssey/waveny/processing"
)

// Main CLI entry point for training.
func Main(arguments []string) error {
	f := newFlags()
	err := f.Parse(arguments)
	if errors.Is(err, flag.ErrHelp) {
		return nil
	}
	if err != nil {
		return err
	}
	return processing.ProcessWithSpagoModel(f.Config, f.SpagoConfig)
}

type flags struct {
	*flag.FlagSet
	processing.Config
	processing.SpagoConfig
}

func newFlags() *flags {
	f := &flags{
		FlagSet: flag.NewFlagSet("waveny process-spago", flag.ContinueOnError),
	}
	f.StringVar(&f.Config.InputPath, "input", "", "Input WAVE file to process.")
	f.StringVar(&f.Config.OutputPath, "output", "", "Output, processed WAVE file.")
	f.StringVar(&f.SpagoConfig.ModelPath, "model", "", "SpaGO model file.")
	return f
}
