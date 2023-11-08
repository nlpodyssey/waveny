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

package live

import (
	"errors"
	"flag"
	"github.com/nlpodyssey/waveny/liveplay"
)

func Main(arguments []string) error {
	f := newFlags()
	err := f.Parse(arguments)
	if errors.Is(err, flag.ErrHelp) {
		return nil
	}
	if err != nil {
		return err
	}
	return liveplay.Run(f.Config)
}

type flags struct {
	*flag.FlagSet
	liveplay.Config
}

func newFlags() *flags {
	f := &flags{
		FlagSet: flag.NewFlagSet("waveny live", flag.ContinueOnError),
	}
	f.StringVar(&f.Config.ModelDataPath, "model", "", "NAM model-data JSON file.")
	f.IntVar(&f.Config.FramesPerBuffer, "fpb", 256, "Frames per buffer.")
	return f
}
