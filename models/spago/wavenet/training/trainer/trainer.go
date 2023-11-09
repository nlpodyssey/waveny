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

package trainer

import (
	"fmt"
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/spago/optimizers"
	"github.com/nlpodyssey/spago/optimizers/adam"
	"github.com/nlpodyssey/waveny/models/spago/wavenet"
	"github.com/nlpodyssey/waveny/models/spago/wavenet/training/datasets"
	"os"
	"path/filepath"
)

type Config struct {
	Model                *wavenet.Model
	TrainingDataLoader   *datasets.DataLoader
	ValidationDataLoader *datasets.DataLoader
	RootDirPath          string
	MaxEpochs            int
	OptimizerLR          float32
}

type Trainer struct {
	model                *wavenet.Model
	adam                 *adam.Adam
	optimizer            *optimizers.Optimizer
	trainingDataLoader   *datasets.DataLoader
	validationDataLoader *datasets.DataLoader
	maxEpochs            int
	rootDirPath          string
	// TODO: implement LR scheduler
}

func New(config Config) *Trainer {
	optimizerStrategy := adam.New(makeAdamConfig(config))
	return &Trainer{
		model:                config.Model,
		adam:                 optimizerStrategy,
		optimizer:            optimizers.New(nn.Parameters(config.Model), optimizerStrategy),
		trainingDataLoader:   config.TrainingDataLoader,
		validationDataLoader: config.ValidationDataLoader,
		maxEpochs:            config.MaxEpochs,
		rootDirPath:          config.RootDirPath,
	}
}

func makeAdamConfig(config Config) adam.Config {
	c := adam.NewDefaultConfig()
	c.StepSize = float64(config.OptimizerLR) // TODO: are StepSize and LR the same thing?
	return c
}

func (t *Trainer) Run() {
	t.model.ResetParameters()
	nn.ZeroGrad(t.model)

	for epoch := 0; epoch < t.maxEpochs; epoch++ {
		t.runEpoch(epoch)
	}
}

func (t *Trainer) runEpoch(epoch int) {
	logf("\nEpoch %d\n", epoch)
	t.trainBatches()
	log("\n")
	mseLoss, esrLoss := t.validate()
	t.saveCheckpoint(epoch, mseLoss, esrLoss)
	log("\n")
}

func (t *Trainer) trainBatches() {
	i := 1
	t.trainingDataLoader.Iterate(func(batch datasets.XYDataPair) {
		if i%10 == 0 {
			log("+")
		} else {
			log(".")
		}
		i++

		loss := t.model.TrainingStep(batch)
		if err := ag.Backward(loss); err != nil {
			panic(fmt.Errorf("backpropagation failed: %w", err))
		}

		if err := t.optimizer.Optimize(); err != nil {
			panic(fmt.Errorf("optimization failed: %w", err))
		}

		t.adam.IncExample() // TODO: correct?
	})
}

func (t *Trainer) validate() (maxMSELoss, maxESRLoss float32) {
	i := 0
	t.validationDataLoader.Iterate(func(batch datasets.XYDataPair) {
		mseLoss, esrLoss := t.model.ValidationStep(batch)

		logf("✓ (%d) MSE=%g ESR=%g\n", i, mseLoss, esrLoss)
		if mseLoss > maxMSELoss || i == 0 {
			maxMSELoss = mseLoss
		}
		if esrLoss > maxESRLoss || i == 0 {
			maxESRLoss = esrLoss
		}

		i++
	})
	return
}

func (t *Trainer) saveCheckpoint(epoch int, mseLoss, esrLoss float32) {
	name := fmt.Sprintf("checkpoint-epoch-%04d-mse-%g-esr-%g", epoch, mseLoss, esrLoss)
	logf("🖫 %s", name)

	base := filepath.Join(t.rootDirPath, name)
	if err := nn.DumpToFile(t.model, base+".spago"); err != nil {
		panic(fmt.Errorf("failed to dump model: %w", err))
	}
	if err := t.model.ExportModelDataFile(base + ".nam"); err != nil {
		panic(fmt.Errorf("failed to dump model: %w", err))
	}
}

func log(a ...any) {
	fmt.Print(a...)
	_ = os.Stdout.Sync()
}

func logf(format string, a ...any) {
	fmt.Printf(format, a...)
	_ = os.Stdout.Sync()
}
