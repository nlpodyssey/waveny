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

package cli

import (
	"fmt"
	"github.com/nlpodyssey/waveny/cli/live"
	"github.com/nlpodyssey/waveny/cli/process_rt"
	"github.com/nlpodyssey/waveny/cli/process_spago"
	"github.com/nlpodyssey/waveny/cli/process_torch"
	"github.com/nlpodyssey/waveny/cli/train"
)

// Main is Waveny command line entry point.
//
// The arguments must NOT include leading program name.
//
// An example invocation:
//
//	cli.Main(os.Args[1:])
func Main(arguments []string) error {
	if len(arguments) == 0 {
		return fmt.Errorf("command argument is missing\n\n%s", usage)
	}

	command := arguments[0]
	arguments = arguments[1:]

	switch command {
	case "help", "-help", "--help", "-h":
		fmt.Print(usage)
		return nil
	case "train":
		return train.Main(arguments)
	case "process-spago":
		return process_spago.Main(arguments)
	case "process-rt":
		return process_rt.Main(arguments)
	case "process-torch":
		return process_torch.Main(arguments)
	case "live":
		return live.Main(arguments)
	default:
		return fmt.Errorf("invalid command\n\n%s", usage)
	}
}
