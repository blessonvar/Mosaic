import time 
import heapq 
import torch 
import torch.nn as nn 
from .sparsegpt import SparseGPT 
from .layerwrapper import WrappedGPT
from .data import get_loaders
import numpy as np
from pdb import set_trace as st 
from collections import defaultdict


def prepare_calibration_input_opt(model, dataloader, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    if "OPT" in model.__class__.__name__:
        layers=model.model.decoder.layers
        
    else:
        layers = model.model.layers

    # dev = model.hf_device_map["model.embed_tokens"]
    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((128, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None,}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass 
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    model.config.use_cache = use_cache
    
    position_ids=None

    return inps, outs, attention_mask, position_ids 




def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def check_sparsity(model):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    layers = model.model.layers
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)  # Assuming find_layers is a function that extracts certain layers or parameters from the given layer

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data  # Accessing the weight parameter of the subset item
            zeros = (W == 0).sum().item()  # Counting zero elements in the weight matrix
            numel = W.numel()  # Total number of elements in the weight matrix

            count += zeros  # Updating total count of zeros for the model
            total_params += numel  # Updating total number of parameters for the model

            sub_count += zeros  # Updating count of zeros for the current layer
            sub_params += numel  # Updating number of parameters for the current layer

            item_sparsity = float(zeros) / numel  # Calculating sparsity for the current item
            print(f"  item '{name}' sparsity: {item_sparsity:.6f}")  # Printing sparsity of the current item

        # After looping through all items in the subset, print the layer sparsity
        layer_sparsity = float(sub_count) / sub_params if sub_params else 0  # Avoid division by zero
        print(f"Layer {i} sparsity: {layer_sparsity:.6f}")

    model.config.use_cache = use_cache
    return float(count) / total_params if total_params else 0  # Return overall sparsity, avoiding division by zero


def check_sparsity_mask(mask):


    W = mask
    count = 0 
    total_params = 0
    count += (W!=0).sum().item()
    total_params += W.numel()



    print(f" density {float(count)/total_params:.6f}")



def check_outlier(mask,threshold):


    W = mask
    count = 0 
    total_params = 0
    
    max_shred=torch.max(W)*threshold
    count += (W>max_shred).sum().item()
    total_params += W.numel()



    outlier_ratio=float(count)/total_params*100
    
    return outlier_ratio


def check_outlier_mean(mask,threshold):


    W = mask
    count = 0 
    total_params = 0
    
    max_shred=torch.mean(W)*threshold
    count += (W>max_shred).sum().item()
    total_params += W.numel()



    outlier_ratio=float(count)/total_params*100
    
    return outlier_ratio


def prepare_calibration_input(model, dataloader, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    # dev = model.hf_device_map["model.embed_tokens"]
    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((128, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass 
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids 

def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha 
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask==True).sum() / W_mask.numel()
    return W_mask, cur_sparsity

def prune_magnitude(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    
    print ("model",model)
    
    layers = model.model.layers
    
    print (layers)
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            W = subset[name].weight.data 
            W_metric = torch.abs(W)
            if prune_n != 0:
                W_mask = (torch.zeros_like(W)==1)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*args.sparsity_ratio)].cpu()
                W_mask = (W_metric<=thresh)

            W[W_mask] = 0
            







def prune_wanda_outlier_structure_special(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    ##### calucalte outlier ratio
    all_layer_ratio=[]
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples, seed=args.seed, seqlen=2048, tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        if "OPT" in model.__class__.__name__:
            print('Experiments with OPT models')
            inps, outs, attention_mask, position_ids = prepare_calibration_input_opt(model, dataloader, device)
        else:
            inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)

    if "opt" in args.model:
        layers=model.model.decoder.layers
    else:
        layers = model.model.layers

    for i in range(len(layers)):
        layer = layers[i]

        subset = find_layers(layer)
        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()
        layer_wmetric=[]

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            activation_data=torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            # layer_wmetric.append(activation_data)
            

            layer_wmetric.append(W_metric)    

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps

        layer_wmetric = torch.cat([torch.flatten(x.cpu()) for x in layer_wmetric])



        for out_ratio in [args.Hyper_m]:
            out_ratio_layer = check_outlier_mean(layer_wmetric, out_ratio)
            print ("layer outlier ratio",out_ratio,out_ratio_layer)

        all_layer_ratio.append(out_ratio_layer)

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()

    print ("before adjustment", all_layer_ratio)

    all_layer_ratio=np.array(all_layer_ratio)
    all_layer_ratio = ((all_layer_ratio - all_layer_ratio.min()) * (1/(all_layer_ratio.max() - all_layer_ratio.min()) * args.Lamda))
    all_layer_ratio=all_layer_ratio-np.mean(all_layer_ratio)
    

    all_layer_ratio=np.round(all_layer_ratio)



 

    for i in range(len(all_layer_ratio)):
        if all_layer_ratio[i] ==1.0:
            all_layer_ratio[i]=2.0
    
    all_layer_ratio=prune_n-all_layer_ratio


    

    print ("after adjustment", all_layer_ratio)
    ############## prune
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=2048,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        if "OPT" in model.__class__.__name__:
            inps, outs, attention_mask, position_ids = prepare_calibration_input_opt(model, dataloader, device)
        else:
            inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)

    # print ("inps",inps)
    if "opt" in args.model:
        layers=model.model.decoder.layers
    else:
        layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        prune_n = int(all_layer_ratio [i])
        print('Layer {} prune_n {} prune_m {}'.format(i, prune_n, prune_m))

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            activation_data=torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            layer_sparsity_ratio= 1-all_layer_ratio[i]
            if layer_sparsity_ratio<=0:
                layer_sparsity_ratio=0.01

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)

            # print ("W_mask",W_mask)
            subset[name].weight.data[W_mask] = 0  ## set weights to zero 

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()

    

def prune_wanda(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=2048,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)



    print ("inps",inps)
    layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if args.use_variant:
                    # wanda variant 
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - args.sparsity_ratio)>0.001) and (alpha_hist[1]-alpha_hist[0]>=0.001):
                        if cur_sparsity > args.sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new 
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # unstructured pruning
                    indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)
#             print ("W_mask",W_mask)
            subset[name].weight.data[W_mask] = 0  ## set weights to zero 

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()







def prune_mag_outlier(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    ##### calucalte outlier ratio
    
    
    
    all_layer_ratio=[]
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=2048,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        
        if "OPT" in model.__class__.__name__:
            
            inps, outs, attention_mask, position_ids = prepare_calibration_input_opt(model, dataloader, device)
        else:
            
            inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)



    print ("inps",inps)
    if "opt" in args.model:
        layers=model.model.decoder.layers
        
    else:
        layers = model.model.layers


    for i in range(len(layers)):
        layer = layers[i]

        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()
            
            
        layer_wmetric=[]

        for name in subset:
            


            

            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))


            activation_data=torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            
            
            
 
            layer_wmetric.append(W_metric)    
                

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps



        layer_wmetric = torch.cat([torch.flatten(x.cpu()) for x in layer_wmetric])
        
        for out_ratio in [args.Hyper_m]:
            
            out_ratio_layer=check_outlier_mean(layer_wmetric,out_ratio)
            print ("layer outlier ratio",out_ratio,out_ratio_layer)

        
        all_layer_ratio.append(out_ratio_layer)
        
        


    print ("before adjustment",all_layer_ratio)

    

    
    
    all_layer_ratio=np.array(all_layer_ratio)
    
    all_layer_ratio = ((all_layer_ratio - all_layer_ratio.min()) * (1/(all_layer_ratio.max() - all_layer_ratio.min()) * args.Lamda*2))
    
    all_layer_ratio=all_layer_ratio-np.mean(all_layer_ratio)+(1-args.sparsity_ratio)
    
    print (all_layer_ratio,np.mean(all_layer_ratio),np.max(all_layer_ratio),np.min(all_layer_ratio))

   
    
                
        
    
    print ("after adjustment",all_layer_ratio  )
    
    

    
    
    ############## prune


    if "opt" in args.model:
        layers=model.model.decoder.layers
        
    else:
        layers = model.model.layers
    
    print (layers)
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            
            layer_sparsity_ratio= 1-all_layer_ratio[i]
            W = subset[name].weight.data 
            W_metric = torch.abs(W)
            if prune_n != 0:
                W_mask = (torch.zeros_like(W)==1)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*layer_sparsity_ratio)].cpu()
                W_mask = (W_metric<=thresh)

            W[W_mask] = 0
                



def prune_wanda_outlier_structure(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    ##### calucalte outlier ratio
    all_layer_ratio=[]
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples, seed=args.seed, seqlen=2048, tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        if "OPT" in model.__class__.__name__:
            print('Experiments with OPT models')
            inps, outs, attention_mask, position_ids = prepare_calibration_input_opt(model, dataloader, device)
        else:
            inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)

    if "opt" in args.model:
        layers=model.model.decoder.layers
    else:
        layers = model.model.layers

    for i in range(len(layers)):
        layer = layers[i]

        subset = find_layers(layer)
        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()
        layer_wmetric=[]

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            activation_data=torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            # layer_wmetric.append(activation_data)
            

            layer_wmetric.append(W_metric)    

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps

        layer_wmetric = torch.cat([torch.flatten(x.cpu()) for x in layer_wmetric])
        
        for out_ratio in [args.Hyper_m]:
            out_ratio_layer = check_outlier_mean(layer_wmetric, out_ratio)
            print ("layer outlier ratio",out_ratio,out_ratio_layer)

        all_layer_ratio.append(out_ratio_layer)

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()

    print ("before adjustment", all_layer_ratio)

    all_layer_ratio=np.array(all_layer_ratio)
    all_layer_ratio = ((all_layer_ratio - all_layer_ratio.min()) * (1/(all_layer_ratio.max() - all_layer_ratio.min()) * args.Lamda))
    all_layer_ratio=all_layer_ratio-np.mean(all_layer_ratio)

    all_layer_ratio=np.round(all_layer_ratio)

    
    all_layer_ratio=prune_n-all_layer_ratio
    

    print ("after adjustment", all_layer_ratio)
    ############## prune
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=2048,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        if "OPT" in model.__class__.__name__:
            inps, outs, attention_mask, position_ids = prepare_calibration_input_opt(model, dataloader, device)
        else:
            inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)

    # print ("inps",inps)
    if "opt" in args.model:
        layers=model.model.decoder.layers
    else:
        layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        prune_n = int(all_layer_ratio [i])
        print('Layer {} prune_n {} prune_m {}'.format(i, prune_n, prune_m))

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            activation_data=torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            layer_sparsity_ratio= 1-all_layer_ratio[i]
            if layer_sparsity_ratio<=0:
                layer_sparsity_ratio=0.01

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
                        
            # print ("W_mask",W_mask)
            subset[name].weight.data[W_mask] = 0  ## set weights to zero 

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()

def prune_wanda_outlier(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    ##### calucalte outlier ratio
    
    
    
    all_layer_ratio=[]
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=2048,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        
        if "OPT" in model.__class__.__name__:
            
            inps, outs, attention_mask, position_ids = prepare_calibration_input_opt(model, dataloader, device)
        else:
            
            inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)



    print ("inps",inps)
    if "opt" in args.model:
        layers=model.model.decoder.layers
        
    else:
        layers = model.model.layers


    for i in range(len(layers)):
        layer = layers[i]

        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()
            
            
        layer_wmetric=[]

        for name in subset:
            


            

            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))


            activation_data=torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            layer_wmetric.append(W_metric)    
                

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps





        layer_wmetric = torch.cat([torch.flatten(x.cpu()) for x in layer_wmetric])
        
        for out_ratio in [args.Hyper_m]:
            
            out_ratio_layer=check_outlier_mean(layer_wmetric,out_ratio)
            print ("layer outlier ratio",out_ratio,out_ratio_layer)

        
        all_layer_ratio.append(out_ratio_layer)
        
        


    print ("before adjustment",all_layer_ratio)

    

    
    
    all_layer_ratio=np.array(all_layer_ratio)
    
    all_layer_ratio = ((all_layer_ratio - all_layer_ratio.min()) * (1/(all_layer_ratio.max() - all_layer_ratio.min()) * args.Lamda*2))
    
    all_layer_ratio=all_layer_ratio-np.mean(all_layer_ratio)+(1-args.sparsity_ratio)
    
    print (all_layer_ratio,np.mean(all_layer_ratio),np.max(all_layer_ratio),np.min(all_layer_ratio))

   
    
                
        
    
    print ("after adjustment",all_layer_ratio  )
    


    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()
    ############## prune
    
    
    
    
    
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=2048,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        
        if "OPT" in model.__class__.__name__:
            
            inps, outs, attention_mask, position_ids = prepare_calibration_input_opt(model, dataloader, device)
        else:
            
            inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)



    print ("inps",inps)
    if "opt" in args.model:
        layers=model.model.decoder.layers
        
    else:
        layers = model.model.layers


    for i in range(len(layers)):
        layer = layers[i]

        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()
            
            
        

        for name in subset:
            

            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))


            activation_data=torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            layer_sparsity_ratio= 1-all_layer_ratio[i]
            
            
            if layer_sparsity_ratio<=0:
                layer_sparsity_ratio=0.01

            print(layer_sparsity_ratio)

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if args.use_variant:
                    # wanda variant 
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - layer_sparsity_ratio)>0.001) and (alpha_hist[1]-alpha_hist[0]>=0.001):
                        if cur_sparsity > layer_sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new 
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # unstructured pruning
                    indices = sort_res[1][:,:int(W_metric.shape[1]*layer_sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)
#             print ("W_mask",W_mask)
            subset[name].weight.data[W_mask] = 0  ## set weights to zero 

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps





    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()





@torch.no_grad()
def prune_sparsegpt(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=2048,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')
    time.sleep(5)
    start_time = time.time()

    for i in range(len(layers)):
        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name, args.sparsity_ratio)
            print('Pruning ...')

            gpts[name].fasterprune(args.sparsity_ratio, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer 
        torch.cuda.empty_cache()

        inps, outs = outs, inps
    end_time = time.time()  # Stops the timer
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")
    time.sleep(5)

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()






@torch.no_grad()
def prune_sparsegpt_mosaic(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    ##### calucalte outlier ratio
    
    device=dev
    
    all_layer_ratio=[]
    mha_layer_ratio=[]
    mlp_layer_ratio=[]
    q_layer_ratio = []
    k_layer_ratio = []
    v_layer_ratio = []
    o_layer_ratio = []
    g_layer_ratio = []
    d_layer_ratio = []
    u_layer_ratio = []

    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    print("loading calibdation data")

    
    

    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=2048,tokenizer=tokenizer)
    print("dataset loading complete")

    time.sleep(5)
    start_time = time.time()

    with torch.no_grad():
        
        if "OPT" in model.__class__.__name__:
            
            inps, outs, attention_mask, position_ids = prepare_calibration_input_opt(model, dataloader, device)
        else:
            
            inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)



    print ("inps",inps)
    if "opt" in args.model:
        layers=model.model.decoder.layers
        
    else:
        layers = model.model.layers

    print(model.hf_device_map)

    for i in range(len(layers)):
        layer = layers[i]

        subset = find_layers(layer)

#        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
#            dev = model.hf_device_map[f"model.layers.{i}"]
#            print("dev:", dev)
#            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()
            
            
        layer_wmetric=[]
        mha_layer_wmetric = []
        mlp_layer_wmetric = []
        q_layer_wmetric = []
        k_layer_wmetric = []
        v_layer_wmetric = []
        o_layer_wmetric = []
        g_layer_wmetric = []
        d_layer_wmetric = []
        u_layer_wmetric = []

        for name in subset:
            print(name)


            

            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))


            activation_data=torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            #if i > 0:
            #    layers[i-1].to('cpu')
            #    layer.to('cuda:0')

            #print(W_metric)

            if name.startswith("self_attn."):
                mha_layer_wmetric.append(W_metric)
                if name.startswith("self_attn.q"):
                    q_layer_wmetric.append(W_metric)
                elif  name.startswith("self_attn.k"):
                    k_layer_wmetric.append(W_metric)
                elif name.startswith("self_attn.v"):
                    v_layer_wmetric.append(W_metric)
                elif name.startswith("self_attn.o"):
                    o_layer_wmetric.append(W_metric)
            elif name.startswith("mlp."):
                mlp_layer_wmetric.append(W_metric)
                if name.startswith("mlp.g"):
                    g_layer_wmetric.append(W_metric)
                elif name.startswith("mlp.d"):
                    d_layer_wmetric.append(W_metric)
                elif name.startswith("mlp.u"):
                    u_layer_wmetric.append(W_metric)

            layer_wmetric.append(W_metric)
            
                

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps

        print("layer_wmetric:", layer_wmetric)
        print("List comprehension output:", [torch.flatten(x.cpu()) for x in layer_wmetric])

        print("MHA layer_wmetric:", mha_layer_wmetric)
        print("MHA List comprehension output:", [torch.flatten(x.cpu()) for x in mha_layer_wmetric])

        print("MHA q layer_wmetric:", q_layer_wmetric)
        print("MHA q List comprehension output:", [torch.flatten(x.cpu()) for x in q_layer_wmetric])

        print("MHA k layer_wmetric:", k_layer_wmetric)
        print("MHA k List comprehension output:", [torch.flatten(x.cpu()) for x in k_layer_wmetric])

        print("MHA v layer_wmetric:", v_layer_wmetric)
        print("MHA v List comprehension output:", [torch.flatten(x.cpu()) for x in v_layer_wmetric])

        print("MHA o layer_wmetric:", o_layer_wmetric)
        print("MHA o List comprehension output:", [torch.flatten(x.cpu()) for x in o_layer_wmetric])

        print("MLP layer_wmetric:", mlp_layer_wmetric)
        print("MLP List comprehension output:", [torch.flatten(x.cpu()) for x in mlp_layer_wmetric])

        print("MLP g layer_wmetric:", g_layer_wmetric)
        print("MLP g List comprehension output:", [torch.flatten(x.cpu()) for x in g_layer_wmetric])

        print("MLP d layer_wmetric:", d_layer_wmetric)
        print("MLP d List comprehension output:", [torch.flatten(x.cpu()) for x in d_layer_wmetric])

        print("MLP u layer_wmetric:", u_layer_wmetric)
        print("MLP u List comprehension output:", [torch.flatten(x.cpu()) for x in u_layer_wmetric])


        def metric_cat(wmetric):
            if wmetric:
                return torch.cat([torch.flatten(x.cpu()) for x in wmetric])
            else:
                default_shape = [1]
                return torch.zeros(default_shape)

        layer_wmetric=metric_cat(layer_wmetric)
        mha_layer_wmetric = metric_cat(mha_layer_wmetric)
        mlp_layer_wmetric = metric_cat(mlp_layer_wmetric)
        q_layer_wmetric = metric_cat(q_layer_wmetric)
        k_layer_wmetric = metric_cat(k_layer_wmetric)
        v_layer_wmetric = metric_cat(v_layer_wmetric)
        o_layer_wmetric = metric_cat(o_layer_wmetric)
        g_layer_wmetric = metric_cat(g_layer_wmetric)
        d_layer_wmetric = metric_cat(d_layer_wmetric)
        u_layer_wmetric = metric_cat(u_layer_wmetric)

        def calc_ratio(wmetric, args):
            for out_ratio in [args.Hyper_m]:
                out_ratio_layer = check_outlier_mean(wmetric, out_ratio)
                #print("layer outlier ratio", out_ratio, out_ratio_layer)
            return out_ratio_layer

        all_layer_ratio.append(calc_ratio(layer_wmetric, args))
        mha_layer_ratio.append(calc_ratio(mha_layer_wmetric, args))
        mlp_layer_ratio.append(calc_ratio(mlp_layer_wmetric, args))
        q_layer_ratio.append(calc_ratio(q_layer_wmetric, args))
        k_layer_ratio.append(calc_ratio(k_layer_wmetric, args))
        v_layer_ratio.append(calc_ratio(v_layer_wmetric, args))
        o_layer_ratio.append(calc_ratio(o_layer_wmetric, args))
        g_layer_ratio.append(calc_ratio(g_layer_wmetric, args))
        d_layer_ratio.append(calc_ratio(d_layer_wmetric, args))
        u_layer_ratio.append(calc_ratio(u_layer_wmetric, args))

    print("before adjustment",all_layer_ratio)
    print("MHA before adjustment", mha_layer_ratio)
    print("MHA q before adjustment", q_layer_ratio)
    print("MHA k before adjustment", k_layer_ratio)
    print("MHA v before adjustment", v_layer_ratio)
    print("MHA o before adjustment", o_layer_ratio)
    print("MLP before adjustment", mlp_layer_ratio)
    print("MLP g before adjustment", g_layer_ratio)
    print("MLP d before adjustment", d_layer_ratio)
    print("MLP u before adjustment", u_layer_ratio)

    all_layer_ratio=np.array(all_layer_ratio)
    
    all_layer_ratio_c = ((all_layer_ratio - all_layer_ratio.min()) * (1/(all_layer_ratio.max() - all_layer_ratio.min()) * args.Lamda*2))
    
    all_layer_ratio=all_layer_ratio_c-np.mean(all_layer_ratio_c)+(1-args.sparsity_ratio)
    
    print (all_layer_ratio,np.mean(all_layer_ratio),np.max(all_layer_ratio),np.min(all_layer_ratio))


    def adjust_ratio(ratio, all_c, args):
        ratio = np.array(ratio)
        ratio = ((ratio - ratio.min()) * (
                1 / (ratio.max() - ratio.min()) * args.Lamda * 2))
        ratio = all_c - np.mean(ratio) + (1 - args.sparsity_ratio)
        print(ratio, np.mean(ratio), np.max(ratio), np.min(ratio))
        return ratio

    mha_layer_ratio = adjust_ratio(mha_layer_ratio, all_layer_ratio_c, args)
    q_layer_ratio = adjust_ratio(q_layer_ratio, all_layer_ratio_c, args)
    k_layer_ratio = adjust_ratio(k_layer_ratio, all_layer_ratio_c, args)
    v_layer_ratio = adjust_ratio(v_layer_ratio, all_layer_ratio_c, args)
    o_layer_ratio = adjust_ratio(o_layer_ratio, all_layer_ratio_c, args)
    mlp_layer_ratio = adjust_ratio(mlp_layer_ratio, all_layer_ratio_c, args)
    g_layer_ratio = adjust_ratio(g_layer_ratio, all_layer_ratio_c, args)
    d_layer_ratio = adjust_ratio(d_layer_ratio, all_layer_ratio_c, args)
    u_layer_ratio = adjust_ratio(u_layer_ratio, all_layer_ratio_c, args)

    print("after adjustment",all_layer_ratio)
    print("MHA after adjustment", mha_layer_ratio)
    print("MHA q after adjustment", q_layer_ratio)
    print("MHA k after adjustment", k_layer_ratio)
    print("MHA v after adjustment", v_layer_ratio)
    print("MHA o after adjustment", o_layer_ratio)
    print("MLP after adjustment", mlp_layer_ratio)
    print("MLP g after adjustment", g_layer_ratio)
    print("MLP d after adjustment", d_layer_ratio)
    print("MLP u after adjustment", u_layer_ratio)

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()
    ############## prune
    print('Starting ...')


    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=2048,tokenizer=tokenizer)





    use_cache = model.config.use_cache
    model.config.use_cache = False
    if "opt" in args.model:
        layers=model.model.decoder.layers
        
    else:
        layers = model.model.layers

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    for i in range(len(layers)):
        layer_sparsity_ratio= 1-all_layer_ratio[i]
        mha_sparsity_ratio=1-mha_layer_ratio[i]
        q_sparsity_ratio = 1 - q_layer_ratio[i]
        k_sparsity_ratio = 1 - k_layer_ratio[i]
        v_sparsity_ratio = 1 - v_layer_ratio[i]
        o_sparsity_ratio = 1 - o_layer_ratio[i]
        mlp_sparsity_ratio = 1-mlp_layer_ratio[i]
        g_sparsity_ratio = 1 - g_layer_ratio[i]
        d_sparsity_ratio = 1 - d_layer_ratio[i]
        u_sparsity_ratio = 1 - u_layer_ratio[i]

        if layer_sparsity_ratio<=0:
            layer_sparsity_ratio=0.01

        if mha_sparsity_ratio<=0:
            mha_sparsity_ratio=0.01

        if q_sparsity_ratio<=0:
            q_sparsity_ratio=0.01

        if k_sparsity_ratio<=0:
            k_sparsity_ratio=0.01

        if v_sparsity_ratio <= 0:
            v_sparsity_ratio = 0.01

        if o_sparsity_ratio <= 0:
            o_sparsity_ratio = 0.01

        if mlp_sparsity_ratio<=0:
            mlp_sparsity_ratio=0.01

        if g_sparsity_ratio<=0:
            g_sparsity_ratio=0.01

        if d_sparsity_ratio<=0:
            d_sparsity_ratio=0.01

        if u_sparsity_ratio <= 0:
            u_sparsity_ratio = 0.01


        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)
            # inps, outs, attention_mask = inps.to(dev), outs.to(dev), attention_mask.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            # outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        for h in handles:
            h.remove()

        for name in gpts:

            sparsity_ratio = 0

            lp = False
            cp = False
            scp = True

            if not lp:
                if name.startswith("self_attn."):
                    if cp:
                        sparsity_ratio = mha_sparsity_ratio
                    else:
                        if name.startswith("self_attn.q"):
                            sparsity_ratio = q_sparsity_ratio
                        elif name.startswith("self_attn.k"):
                            sparsity_ratio = k_sparsity_ratio
                        elif name.startswith("self_attn.v"):
                            sparsity_ratio = v_sparsity_ratio
                        elif name.startswith("self_attn.o"):
                            sparsity_ratio = o_sparsity_ratio
                elif name.startswith("mlp."):
                    if cp:
                        sparsity_ratio = mlp_sparsity_ratio
                    else:
                        if name.startswith("mlp.g"):
                            sparsity_ratio = g_sparsity_ratio
                        elif name.startswith("mlp.d"):
                            sparsity_ratio = d_sparsity_ratio
                        elif name.startswith("mlp.u"):
                            sparsity_ratio = u_sparsity_ratio
            else:
                sparsity_ratio = layer_sparsity_ratio

            print(i, name, sparsity_ratio)
            print('Pruning ...')

            gpts[name].fasterprune(sparsity_ratio, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            # outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        layers[i] = layer 
        torch.cuda.empty_cache()

        inps, outs = outs, inps


    end_time = time.time()  # Stops the timer
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()



@torch.no_grad()
def prune_sparsegpt_outlier(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    ##### calucalte outlier ratio
    
    device=dev
    
    all_layer_ratio=[]
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=2048,tokenizer=tokenizer)
    print("dataset loading complete")
    
    time.sleep(5)
    start_time = time.time()


    with torch.no_grad():
        
        if "OPT" in model.__class__.__name__:
            
            inps, outs, attention_mask, position_ids = prepare_calibration_input_opt(model, dataloader, device)
        else:
            
            inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)



    print ("inps",inps)
    if "opt" in args.model:
        layers=model.model.decoder.layers
        
    else:
        layers = model.model.layers


    for i in range(len(layers)):
        layer = layers[i]

        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()
            
            
        layer_wmetric=[]

        for name in subset:
            


            

            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))


            activation_data=torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            layer_wmetric.append(W_metric)
            
                

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps





        layer_wmetric = torch.cat([torch.flatten(x.cpu()) for x in layer_wmetric])
        
        for out_ratio in [args.Hyper_m]:
            
            out_ratio_layer=check_outlier_mean(layer_wmetric,out_ratio)
            print ("layer outlier ratio",out_ratio,out_ratio_layer)

        
        all_layer_ratio.append(out_ratio_layer)
        




    print ("before adjustment",all_layer_ratio)



    
    
    all_layer_ratio=np.array(all_layer_ratio)
    
    all_layer_ratio = ((all_layer_ratio - all_layer_ratio.min()) * (1/(all_layer_ratio.max() - all_layer_ratio.min()) * args.Lamda*2))
    
    all_layer_ratio=all_layer_ratio-np.mean(all_layer_ratio)+(1-args.sparsity_ratio)
    
    print (all_layer_ratio,np.mean(all_layer_ratio),np.max(all_layer_ratio),np.min(all_layer_ratio))

   
 
    
                
        
    
    
    print ("after adjustment",all_layer_ratio  )
    
    



    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()
    ############## prune
    print('Starting ...')


    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=2048,tokenizer=tokenizer)





    use_cache = model.config.use_cache
    model.config.use_cache = False
    if "opt" in args.model:
        layers=model.model.decoder.layers
        
    else:
        layers = model.model.layers

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    for i in range(len(layers)):



        layer_sparsity_ratio= 1-all_layer_ratio[i]
        
        
        if layer_sparsity_ratio<=0:
            layer_sparsity_ratio=0.01


        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)
            # inps, outs, attention_mask = inps.to(dev), outs.to(dev), attention_mask.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            # outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            gpts[name].fasterprune(layer_sparsity_ratio, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            # outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        layers[i] = layer 
        torch.cuda.empty_cache()

        inps, outs = outs, inps




    end_time = time.time()  # Stops the timer
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")



    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
