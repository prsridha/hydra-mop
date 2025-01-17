# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from hydra.utilities import delete_batch, move_batch_to_device
import torch

"""
    Generic Forward pass module. Must take as input the model, device, and batch.
    Returns output batch.

"""

class Forward():
    def __init__(self, idx):
        self.type="Forward"
        self.idx = idx

    def run(self, model, batch_input, device):
        model.to(device, non_blocking=True)

        batch_input = move_batch_to_device(batch_input, device)
        try:
            print("F: {}".format(batch_input.shape))
        except:
            pass
        
        with torch.no_grad() and torch.cuda.amp.autocast():
            ns_labels = model(batch_input)

        delete_batch(batch_input)
            
        return ns_labels
