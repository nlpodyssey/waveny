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

package train

import (
	"errors"
	"flag"
	"github.com/nlpodyssey/waveny/models/spago/wavenet/training"
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
	return training.Train(f.PathsConfig, f.Config)
}

type flags struct {
	*flag.FlagSet
	training.PathsConfig
	training.Config
}

func newFlags() *flags {
	f := &flags{
		FlagSet: flag.NewFlagSet("waveny train", flag.ContinueOnError),
	}

	f.StringVar(&f.PathsConfig.ConfigPath, "config", "", "Model configuration JSON file.")
	f.StringVar(&f.PathsConfig.InputPath, "input", "", "Clean audio file.")
	f.StringVar(&f.PathsConfig.TargetPath, "target", "", "Target (reamped) audio file.")
	f.StringVar(&f.PathsConfig.OutDirPath, "out", "", "Output directory for model checkpoints.")

	f.IntVar(&f.Config.MaxEpochs, "epochs", 100, "Maximum training epochs.")

	f.IntVar(&f.Config.TrainingSetStart, "ts-start", 0, "Training set split start.")
	f.IntVar(&f.Config.TrainingSetStop, "ts-stop", -432000, "Training set split stop.")
	f.IntVar(&f.Config.TrainingSetNY, "ts-ny", 8192, "Training set nx.")

	f.IntVar(&f.Config.ValidationSetStart, "vs-start", -432000, "Validation set split start.")
	f.IntVar(&f.Config.ValidationSetStop, "vs-stop", 0, "Validation set split stop.")
	f.IntVar(&f.Config.ValidationSetNY, "vs-ny", 0, "Validation set nx.")

	f.IntVar(&f.Config.TrainingBatchSize, "td-batch", 16, "Training data-loader batch size.")
	f.BoolVar(&f.Config.TrainingShuffle, "td-shuffle", true, "Training data-loader shuffle flag.")
	f.BoolVar(&f.Config.TrainingDropLast, "td-drop-last", true, "Training data-loader drop-last flag.")

	return f
}
