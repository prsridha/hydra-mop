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


# This task requres A LOT of DRAM, so be careful!

import hydra
from hydra import ModelTask, ModelOrchestrator
import customLayers as custom
import copy
import torch
from torchtext.experimental.datasets import WikiText2
from torch.utils.data import DataLoader
from os import path
from timeit import timeit as timer

"""
    Preprocessing functions for the dataloaders.
"""
def process_raw_data(raw_data, batch_size, bptt):
    _num = raw_data.size(0) // (batch_size * bptt)
    raw_data = raw_data[:(_num * batch_size * bptt)]
    return raw_data

def collate_batch(batch_data, batch_size, mask_frac, mask_id, cls_id):
    #print(type(batch_data))
    batch_data = torch.tensor(batch_data)
    batch_data = batch_data.long()
    batch_data = batch_data.view(batch_size, -1)
    batch_data = batch_data.t().contiguous()
    # Generate masks with args.mask_frac
    data_len = batch_data.size(0)
    ones_num = int(data_len * mask_frac)
    zeros_num = data_len - ones_num
    lm_mask = torch.cat([torch.zeros(zeros_num), torch.ones(ones_num)])
    lm_mask = lm_mask[torch.randperm(data_len)]
    batch_data = torch.cat((torch.tensor([[cls_id] * batch_data.size(1)]).long(), batch_data))
    lm_mask = torch.cat((torch.tensor([0.0]), lm_mask))

    targets = torch.stack([batch_data[i] for i in range(lm_mask.size(0)) if lm_mask[i]]).view(-1)
    batch_data = batch_data.masked_fill(lm_mask.bool().unsqueeze(1), mask_id)
    return batch_data, [lm_mask, targets]

"""
    Custom loss function
"""

def pretraining_loss(out, targets):
    lm_mask, label = targets
    out = torch.stack([out[i] for i in range(lm_mask.size(0)) if lm_mask[i]])
    loss_computer = torch.nn.CrossEntropyLoss()
    out = out.view(-1, 28783)
    label = label.to(out.device)

    return loss_computer(out, label)


"""
    Helper function to create a training dataloader.
"""

def get_data_loader_train(b_size):
    start = timer()
    print("\nPreparing to load dataset....")


    if (path.exists("vocab/torchtext_bert_vocab_wiki.pt")):
        vocab = torch.load("vocab/torchtext_bert_vocab_wiki.pt")
        dataset = WikiText2(vocab=vocab, split='train') # set to train for real testing.pi
    else:
        dataset = WikiText2(split='train') # set to train for real testing.pi
        vocab = dataset.get_vocab()
        torch.save(vocab, "vocab/torchtext_bert_vocab_wiki.pt")
        dataset = WikiText2(vocab = vocab, split='train') # set to train for real testing.pi


    dataset.data = torch.cat(tuple(filter(lambda t: t.numel() > 0, dataset)))

    mask_id = vocab.stoi['<MASK>']
    cls_id = vocab.stoi['<cls>']
    bptt = 128
    mask_frac = 0.15
    end = timer()
    dataset = process_raw_data(dataset.data, b_size, bptt)
    print("Dataset Loaded in {} seconds.".format(end-start))



    return DataLoader(dataset, batch_size=b_size * bptt, shuffle=True,
                                collate_fn=lambda b: collate_batch(b, b_size, mask_frac, mask_id, cls_id), drop_last=True)




"""
    Main function
"""
def main():


    device_count = torch.cuda.device_count()

    model_0 = torch.nn.Sequential(

        custom.BertEmbedding(28783, 1024, transpose=False),
        
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        # 4
        
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        # 8
        
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        # 12
        
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        # 16
        
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        # 20
        
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        # 24 - BERT-Large Size (340M params)
        
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        # 28
        
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        # 32
        
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        # 12
        
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        # 36
        
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        # 40
        
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        # 44
        
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        # 48
        
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        # 52
        
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        # 56
        
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        # 60
        
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        # 64
        
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        # 68
        
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        # 72
        
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        # 76
        
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        # 80
        
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        # 84
        
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        # 88
        
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        # 92
        
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        # 28
        
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        # 96
        
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        # 100
        
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        # 104
        
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        # 108
        
        
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        # 112
        
        
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        # 116
        
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        # 120
        
         
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        # 124
        
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        # 128
        
        
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        # 132
        
        
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        custom.BertTransformerEncoderLayer(1024, 16, 1024, 0.5),
        # 136
       
        torch.nn.Linear(1024, 1024),
        torch.nn.GELU(),
        torch.nn.LayerNorm(1024, eps=1e-12),
        torch.nn.Linear(1024, 28783)

    )
    
    params = sum(p.numel() for p in model_0.parameters())
    print("Total parameters: {}".format(params))

    
    model_1 = copy.deepcopy(model_0)
    model_2 = copy.deepcopy(model_0)
    model_3 = copy.deepcopy(model_0)
    model_4 = copy.deepcopy(model_0)
    model_5 = copy.deepcopy(model_0)
    model_6 = copy.deepcopy(model_0)
    model_7 = copy.deepcopy(model_0)
    model_8 = copy.deepcopy(model_0)
    model_9 = copy.deepcopy(model_0)
    model_10 = copy.deepcopy(model_0)
    model_11 = copy.deepcopy(model_0)
    model_12 = copy.deepcopy(model_0)

    dataloader_0 = get_data_loader_train(8) # Generate dataloader
    dataloader_1 = get_data_loader_train(16)
    dataloader_2 = get_data_loader_train(32)
    
    dataloader_3 = get_data_loader_train(8) # Generate dataloader
    dataloader_4 = get_data_loader_train(16)
    dataloader_5 = get_data_loader_train(32)
    
    dataloader_6 = get_data_loader_train(8) # Generate dataloader
    dataloader_7 = get_data_loader_train(16)
    dataloader_8 = get_data_loader_train(32)
    
    dataloader_9 = get_data_loader_train(8) # Generate dataloader
    dataloader_10 = get_data_loader_train(16)
    dataloader_11 = get_data_loader_train(32)

    
    task_0 = ModelTask("Model 0", model_0, pretraining_loss, dataloader_0, 0.001, 4)
    task_1 = ModelTask("Model 1", model_1, pretraining_loss, dataloader_1, 0.001, 4)
    task_2 = ModelTask("Model 2", model_2, pretraining_loss, dataloader_2, 0.001, 4)
    
    task_3 = ModelTask("Model 3", model_3, pretraining_loss, dataloader_3, 0.0001, 4)
    task_4 = ModelTask("Model 4", model_4, pretraining_loss, dataloader_4, 0.0001, 4)
    task_5 = ModelTask("Model 5", model_5, pretraining_loss, dataloader_5, 0.0001, 4)
    
    task_6 = ModelTask("Model 6", model_6, pretraining_loss, dataloader_6, 0.00001, 4)
    task_7 = ModelTask("Model 7", model_7, pretraining_loss, dataloader_7, 0.00001, 4)
    task_8 = ModelTask("Model 8", model_8, pretraining_loss, dataloader_8, 0.00001, 4)
    
    task_9 = ModelTask("Model 9", model_9, pretraining_loss, dataloader_9, 0.000001, 4)
    task_10 = ModelTask("Model 10", model_10, pretraining_loss, dataloader_10, 0.000001, 4)
    task_11 = ModelTask("Model 11", model_11, pretraining_loss, dataloader_11, 0.000001, 4)


    # create orchestrator
    orchestra = ModelOrchestrator([task_0, task_1, task_2, task_3, task_4, task_5, task_6, task_7, task_8, task_9, task_10, task_11])
    orchestra.verbose = 1

    """
     The Double-Buffer. Adjusting this up or down a bit can help to address minor
     errors in partitioning memory consumption.
    """
    orchestra.buffer = 10000

    orchestra.generate()
    orchestra.train_models()
    
if __name__ == "__main__":
    main()