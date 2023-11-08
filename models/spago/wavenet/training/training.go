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

package training

import (
	"fmt"
	"github.com/nlpodyssey/waveny/models/spago/wavenet"
	datasets2 "github.com/nlpodyssey/waveny/models/spago/wavenet/training/datasets"
	"github.com/nlpodyssey/waveny/models/spago/wavenet/training/trainer"
	"os"
	"path/filepath"
	"time"
)

// Config provides the main settings for training a model.
type Config struct {
	MaxEpochs int // training epochs

	TrainingSetStart int // starting offset for training-set split
	TrainingSetStop  int // ending offset for training-set split
	TrainingSetNY    int

	ValidationSetStart int // starting offset for validation-set split
	ValidationSetStop  int // ending offset for validation-set split
	ValidationSetNY    int

	TrainingBatchSize int  // batch size
	TrainingShuffle   bool // whether to enable shuffling
	TrainingDropLast  bool
}

// PathsConfig provides a series of paths to input/output files or folders.
type PathsConfig struct {
	ConfigPath string // JSON configuration file describing the model to train
	InputPath  string // clean input WAVE file
	TargetPath string // reamped WAVE file
	OutDirPath string // output directory for trained models
}

// Train is the higher-level training procedure. It prepares the datasets
// from the given configurations, then runs the actual training, via the
// (lower-level) trainer.Trainer.
func Train(pathsConfig PathsConfig, config Config) error {
	modelConfig, err := wavenet.ReadModelConfigJSONFile(pathsConfig.ConfigPath)
	if err != nil {
		return fmt.Errorf("failed to read JSON model config from file %q: %w", pathsConfig.ConfigPath, err)
	}

	model := wavenet.New(modelConfig.Net.Config)

	trainingSet, err := datasets2.NewDataset(makeTrainingDatasetConfig(pathsConfig, config, model))
	if err != nil {
		return fmt.Errorf("failed to make training dataset: %w", err)
	}
	validationSet, err := datasets2.NewDataset(makeValidationDatasetConfig(pathsConfig, config, model))
	if err != nil {
		return fmt.Errorf("failed to make validation dataset: %w", err)
	}
	trainingDataLoader := datasets2.NewDataLoader(trainingSet, makeTrainingDataLoaderConfig(config))
	validationDataLoader := datasets2.NewDataLoader(validationSet, validationDataLoaderConfig)

	rootDirPath, err := createRootDirPath(pathsConfig.OutDirPath)
	if err != nil {
		return err
	}

	t := trainer.New(trainer.Config{
		Model:                model,
		TrainingDataLoader:   trainingDataLoader,
		ValidationDataLoader: validationDataLoader,
		RootDirPath:          rootDirPath,
		MaxEpochs:            config.MaxEpochs,
		OptimizerLR:          modelConfig.Optimizer.LR,
	})
	t.Run()

	return nil
}

func makeTrainingDatasetConfig(pathsConfig PathsConfig, config Config, model *wavenet.Model) datasets2.DatasetConfig {
	return datasets2.DatasetConfig{
		XPath: pathsConfig.InputPath,
		YPath: pathsConfig.TargetPath,
		Start: config.TrainingSetStart,
		Stop:  config.TrainingSetStop,
		NX:    model.ReceptiveField,
		NY:    config.TrainingSetNY,
	}
}

func makeValidationDatasetConfig(pathsConfig PathsConfig, config Config, model *wavenet.Model) datasets2.DatasetConfig {
	return datasets2.DatasetConfig{
		XPath: pathsConfig.InputPath,
		YPath: pathsConfig.TargetPath,
		Start: config.ValidationSetStart,
		Stop:  config.ValidationSetStop,
		NX:    model.ReceptiveField,
		NY:    config.ValidationSetNY,
	}
}

func makeTrainingDataLoaderConfig(config Config) datasets2.DataLoaderConfig {
	return datasets2.DataLoaderConfig{
		BatchSize: config.TrainingBatchSize,
		Shuffle:   config.TrainingShuffle,
		DropLast:  config.TrainingDropLast,
	}
}

var validationDataLoaderConfig = datasets2.DataLoaderConfig{
	BatchSize: 1,
	Shuffle:   false,
	DropLast:  false,
}

func createRootDirPath(base string) (string, error) {
	subFolder := time.Now().Format("2006-01-02_15-04-05")
	path := filepath.Join(base, subFolder)
	if err := os.Mkdir(path, 0755); err != nil {
		return "", fmt.Errorf("failed to create training output directory %q: %w", path, err)
	}
	return path, nil
}
