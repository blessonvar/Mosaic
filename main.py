import argparse
import os 
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM, BitsAndBytesConfig
# from importlib.metadata import version
from collections import defaultdict
from lib.prune_all import prune_wanda_outlier_structure_special,prune_wanda_outlier_structure,prune_sparsegpt_outlier,prune_wanda_outlier,prune_mag_outlier, prune_wanda,prune_magnitude,prune_sparsegpt, prune_sparsegpt_outlier, prune_sparsegpt_mosaic, check_sparsity, find_layers
from lib.eval import eval_ppl
import sys
print('# of gpus: ', torch.cuda.device_count())



import json
import logging
import math

import random
from itertools import chain
from pathlib import Path

import datasets
import torch
import torch.nn as nn
import torch.nn.functional as F

from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import check_min_version, get_full_repo_name, send_example_telemetry
from transformers.utils.versions import require_version

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


logger = get_logger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

def get_llm(model, cache_dir="llm_weights", low_mem=False):
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    model_args = {
        "torch_dtype": torch.float16,
        "cache_dir": cache_dir,
        "low_cpu_mem_usage": True,
        "device_map": "auto",
        #"from_tf": True,
        "use_auth_token":True,
        #"max_memory": {0: 20},
        #"offload_folder": "offload_weights"
    }

    if low_mem:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        model_args["quantization_config"] = bnb_config

    model = LlamaForCausalLM.from_pretrained(
        model,
        **model_args
    )
    model.tie_weights()
    model.seqlen = 2048
    print(model.hf_device_map)
    print(model.model.layers[-1].self_attn.q_proj.weight)
    print(model.share_memory)
    return model

def main():


    ########################## for prune ################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--sparsity_ratio', type=float, default=0, help='Sparsity level')
    parser.add_argument("--sparsity_type", type=str)
    parser.add_argument("--prune_method", type=str)
    parser.add_argument("--cache_dir", default="llm_weights", type=str )
    parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant described in the appendix")
    parser.add_argument('--save', type=str, default=None, help='Path to save results.')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
    #parser.add_argument('--sponly', type=str, default=None, help='')
    #parser.add_argument("--low_mem", type=bool, default=False, help='Use 4-bit weights.')


########################################### for train
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="wikitext",
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default="wikitext-2-raw-v1",
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."
        ),
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=32,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--no_keep_linebreaks", action="store_true", help="Do not keep line breaks when using TXT files."
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        default=False,
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )
    parser.add_argument(
        "--low_mem",
        action="store_true",
        help=(
            "."
        ),
    )

    parser.add_argument(
        "--sponly",
        action="store_true",
        help=(
            "."
        ),
    )

    #### saving parameters #####
    
    parser.add_argument(
        "--method",
        type=str,
        default=None,

    )   

    #### data parameters #####
    
    parser.add_argument(
        "--Lamda",
        default=0.08,
        type=float,
        help="Lamda",
    )

    parser.add_argument(
        '--Hyper_m', 
        type=float,
        default=3, )
    
    parser.add_argument(
    "--outlier_by_activation", action="store_true", help="outlier_by_activation")  
    
    
    parser.add_argument(
    "--outlier_by_wmetric", action="store_true", help="outlier_by_wmetric")  

    args = parser.parse_args()

    if not args.sponly:
        print ("args.nsamples",args.nsamples)
        print(f'{int(torch.cuda.mem_get_info()[0]/1024**3)-2}GB')
        # Setting seeds for reproducibility
        np.random.seed(args.seed)
        torch.random.manual_seed(args.seed)

        # Handling n:m sparsity
        prune_n, prune_m = 0, 0
        if args.sparsity_type != "unstructured":
            assert args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
            prune_n, prune_m = map(int, args.sparsity_type.split(":"))

        model_name = args.model.split("/")[-1]
        print(f"loading llm model {args.model}")
        model = get_llm(args.model, args.cache_dir, args.low_mem)

        print ("model is =================================================================================")
        print (model.__class__.__name__)
        print (model)
        print (sum(p.numel() for p in model.parameters() if p.requires_grad))

        model.eval()

        print(args.save_model)

        #if not args.prune_method == "bep":
 #if "opt" in args.model:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
#        elif "llama" in args.model:
 #tokenizer = LlamaTokenizer.from_pretrained(args.model, use_fast=False)
        #else:
        #    tokenizer = None

        device = torch.device("cuda:0")
        if "30b" in args.model or "65b" in args.model: # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
            device = model.hf_device_map["lm_head"]
        print("use device ", device)

        print ("target sparsity", args.sparsity_ratio)
        
        if args.sparsity_ratio > 0.0:
            print("pruning starts")

            ############################ baseline   ############################
            if args.prune_method == "wanda":
                prune_wanda(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
            elif args.prune_method == "magnitude":
                prune_magnitude(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
            elif args.prune_method == "sparsegpt":
                prune_sparsegpt(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
            ############################ owl   ############################
            elif args.prune_method == "wanda_owl":
                prune_wanda_outlier(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
            ############################ owl   ############################
            elif args.prune_method == "magnitude_owl":
                prune_mag_outlier(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
            elif args.prune_method == "sparsegpt_owl":
                prune_sparsegpt_outlier(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
            elif args.prune_method == "sparsegpt_mosaic":
                prune_sparsegpt_mosaic(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
            elif args.prune_method == "wanda_owl_structure":
                prune_wanda_outlier_structure(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
            elif args.prune_method == "wanda_owl_structure_special":
                prune_wanda_outlier_structure_special(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)

        ################################################################
        print("*"*30)
        sparsity_ratio = check_sparsity(model)
        print(f"sparsity sanity check {sparsity_ratio:.4f}")
        print("*"*30)
        ################################################################


        ppl_wikitext2 = eval_ppl(model, tokenizer, device, False, "wikitext2")
        ppl_ptb = eval_ppl(model, tokenizer, device, False, "ptb")

        print(f"ppl on wikitext {ppl_wikitext2}")
        print(f"ppl on ptb {ppl_ptb}")

        results = 'results-'+str(model_name)+'-'+str(args.prune_method)+'-'+str(args.sparsity_ratio)+'.json'
        
        try:
            with open(results, 'r') as file:
                data = json.load(file)
        except FileNotFoundError:
            data = {}

        if args.model in data:
            data[args.model].update({
                args.prune_method: {
                    'sparsity': args.sparsity_ratio,
                    'wikitext2': ppl_wikitext2,
                    'ptb': ppl_ptb
                    }
                })
        else:
            data[args.model] = {
                    args.prune_method: {
                        'sparsity': args.sparsity_ratio,
                        'wikitext2': ppl_wikitext2,
                        'ptb': ppl_ptb
                    }
                }

        with open(results, 'w') as file:
            json.dump(data, file, indent=4)

        print("Results saved")



        sys.stdout.flush()

        # print(f"final ppl on wikitext {ppl}")

        if args.save_model:
            model.save_pretrained(args.save_model)
            tokenizer.save_pretrained(args.save_model)
            
            torch.save({
                        'model': model, 
                        'tokenizer': tokenizer,
                        }, "save_test/model.bin")

            print(f"model saved to {args.save_model}")


    else:
        print("SP Pruning")
        # Load the model
        model = LlamaForCausalLM.from_pretrained(args.save_model)


        # Load the tokenizer
        #print("Loading tokenizer")
        #tokenizer = LlamaForCausalLM.from_pretrained(args.save_model)

        print(model)

        print(model.model.layers)

        q_proj_masked_weights = model.model.layers[0].self_attn.q_proj.weight.data
        k_proj_masked_weights = model.model.layers[0].self_attn.k_proj.weight.data
        #v_proj_masked_weights = model.model.layers[0].self_attn.v_proj.weight.data

        #check_zero_rows_sparsity_and_order(v_proj_masked_weights)

        save_model_sparsity_info(model, "sparsity_data.json")

        #print_sparsity(model)
        #shared_zero_indices = get_shared_masked_zero_indices(model)
        #print(f"Shared zero indices (example showing up to 10): {shared_zero_indices[:10]}")
        #recreate_linear_layers(model, shared_zero_indices)

        print(model)


def save_model_sparsity_info(model, file_name):
    # Initialize a dictionary to hold sparsity info
    sparsity_info = {}

    # Iterate through each named parameter in the model
    for name, parameter in model.named_parameters():
        if parameter.dim() == 2:  # Focus on weight tensors typically of dimension 2
            weights = parameter.data

            # Check if any row is entirely zero
            zero_rows = torch.all(weights == 0, dim=1)

            # Count how many rows are entirely zero
            num_zero_rows = torch.sum(zero_rows).item()

            # Calculate sparsity percentage for each row
            sparsity_percentages = torch.sum(weights == 0, dim=1).float() / weights.size(1) * 100

            # Sort the rows by sparsity percentage in descending order
            sorted_indices = torch.argsort(sparsity_percentages, descending=True)
            sorted_sparsity_percentages = sparsity_percentages[sorted_indices]

            # Build a list of sparsity info for each row
            rows_sparsity_info = [
                {"rank": rank, "row_index": row_index.item() + 1, "sparsity_percentage": sparsity.item()}
                for rank, (row_index, sparsity) in enumerate(zip(sorted_indices, sorted_sparsity_percentages), start=1)
            ]

            # Add the layer's sparsity info to the dictionary
            sparsity_info[name] = {
                "num_zero_rows": num_zero_rows,
                "rows_sparsity_info": rows_sparsity_info
            }

    # Save the sparsity info as JSON
    with open(file_name, 'w') as f:
        json.dump(sparsity_info, f, indent=4)

def check_zero_rows_sparsity_and_order(*weight_tensors):
    for i, weights in enumerate(weight_tensors, start=1):
        # Check if any row is entirely zero
        zero_rows = torch.all(weights == 0, dim=1)

        # Count how many rows are entirely zero
        num_zero_rows = torch.sum(zero_rows).item()

        if num_zero_rows > 0:
            print(f"Tensor {i} has {num_zero_rows} row(s) set to zero.")
        else:
            print(f"Tensor {i} has no rows set to zero.")

        # Calculate sparsity percentage for each row
        sparsity_percentages = torch.sum(weights == 0, dim=1).float() / weights.size(1) * 100

        # Sort the rows by sparsity percentage in descending order
        sorted_indices = torch.argsort(sparsity_percentages, descending=True)
        sorted_sparsity_percentages = sparsity_percentages[sorted_indices]

        # Print sparsity percentage of each row in sorted order
        print(f"Tensor {i} row sparsity from most to least sparse:")
        for rank, (row_index, sparsity) in enumerate(zip(sorted_indices, sorted_sparsity_percentages), start=1):
            print(f"Rank {rank}: Row {row_index.item() + 1} is {sparsity:.2f}% sparse.")

def adjust_linear_layer(layer, shared_zero_indices):
    """
    Adjusts a linear layer by removing the weights corresponding to the shared zero indices.

    Parameters:
    - layer: The linear layer to be adjusted.
    - shared_zero_indices: A list or tensor of indices of the output features to be removed.

    Returns:
    - A new linear layer with the specified output features removed.
    """
    # Convert shared_zero_indices to a tensor of appropriate type if it's not already
    if not isinstance(shared_zero_indices, torch.Tensor):
        shared_zero_indices = torch.tensor(shared_zero_indices, dtype=torch.long)
    else:
        shared_zero_indices = shared_zero_indices.to(dtype=torch.long)

    indices_to_remove = shared_zero_indices[:,1]
    print(indices_to_remove)
    print(len(indices_to_remove))

    # Get the current weight and bias of the layer
    current_weight = layer.weight.data
    current_bias = layer.bias.data if layer.bias is not None else None

    # Determine the indices of the rows to keep
    all_indices = torch.arange(current_weight.size(0), dtype=torch.long)
    mask = ~torch.isin(all_indices, indices_to_remove)
    keep_indices = all_indices[mask]

    # Select the rows (weights) and elements (bias) to keep
    adjusted_weight = current_weight[keep_indices]
    adjusted_bias = current_bias[keep_indices] if current_bias is not None else None

    # Create a new linear layer with the adjusted dimensions
    new_layer = nn.Linear(layer.in_features, adjusted_weight.size(0), bias=layer.bias is not None)
    new_layer.weight.data = adjusted_weight
    if layer.bias is not None:
        new_layer.bias.data = adjusted_bias

    return new_layer


def recreate_linear_layers(model, shared_zero_indices):
    print(len(shared_zero_indices))

    """
    Recreates the q_proj, k_proj, v_proj, and o_proj layers with adjustments for shared zero indices.
    """
    self_attn = model.model.layers[0].self_attn

    # Adjust q_proj, k_proj, and v_proj by removing shared zero indices
    self_attn.q_proj = adjust_linear_layer(self_attn.q_proj, shared_zero_indices)
    self_attn.k_proj = adjust_linear_layer(self_attn.k_proj, shared_zero_indices)
    self_attn.v_proj = adjust_linear_layer(self_attn.v_proj, shared_zero_indices)

    # For o_proj, we need to adjust the input features to match the new output dimension of q, k, v
    new_out_features = self_attn.q_proj.out_features  # Assuming all three have been adjusted equally
    o_proj_adjusted = nn.Linear(new_out_features, self_attn.o_proj.out_features, bias=False)

    # Here we assume we're reducing the dimension based on the output of q_proj, k_proj, v_proj
    # This may require adjusting the way you handle the forward pass in these layers.

    self_attn.o_proj = o_proj_adjusted

def get_shared_masked_zero_indices(model):
    # Access the masked weights of the q_proj, k_proj, v_proj layers in the first decoder layer
    q_proj_masked_weights = model.model.layers[0].self_attn.q_proj.weight.data
    k_proj_masked_weights = model.model.layers[0].self_attn.k_proj.weight.data
    v_proj_masked_weights = model.model.layers[0].self_attn.v_proj.weight.data

    # Create masks where weights are zero
    q_mask = q_proj_masked_weights == 0
    k_mask = k_proj_masked_weights == 0
    v_mask = v_proj_masked_weights == 0

    # Find intersection of masks (positions that are zero in all layers)
    shared_zero_mask = q_mask & k_mask & v_mask

    # Calculate percentage of shared zero weights (for reporting purposes)
    shared_zero_percentage = torch.sum(shared_zero_mask).float() / shared_zero_mask.nelement() * 100
    print(
        f"Percentage of weight indexes that are zero (masked) shared by q, k, v layers in self_attn: {shared_zero_percentage:.2f}%")

    # Extract the indices of shared zeros
    shared_zero_indices = torch.nonzero(shared_zero_mask, as_tuple=False)

    # Convert tensor to a list of tuples for easier interpretation/usage
    shared_zero_indices_list = list(map(tuple, shared_zero_indices.tolist()))

    #print(shared_zero_indices_list)

    return shared_zero_indices_list


def print_sparsity(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weight = module.weight.data
            sparsity = float(torch.sum(weight == 0)) / float(weight.nelement())
            print(f'{name}: {sparsity:.4f}')

if __name__ == '__main__':
    main()

