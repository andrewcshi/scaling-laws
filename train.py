import os
import time
import math
import pickle
from contextlib import nullcontext
import pandas as pd

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

# ---------------------------------------------------------------------
# default config
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'  # Change to 'resume' if you want to load from ckpt.pt
wandb_log = False
wandb_project = 'owt'
wandb_run_name = 'gpt2'
wandb_run_id = ""  # empty unless resuming exactly one run
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8
batch_size = 12
block_size = 1024
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False
learning_rate = 6e-4
max_iters = 600000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5
backend = 'nccl'
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True
scaling = ""
scale_N = False
scale_D = False
estimate_B_crit = False
fraction_of_data = 1.0

D = int(fraction_of_data * 9035582198)
# ---------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read())  # your existing config overrides
config = {k: globals()[k] for k in config_keys}

def get_flop_per_token(num_embd, num_layers):
    return 6 * 12 * num_embd**2 * num_layers

def run_single_train(n_layer_val, n_embd_val, run_idx=0, total_runs=1):
    global n_layer, n_embd, D, out_dir, wandb_run_name
    global n_head

    # override
    n_layer = n_layer_val
    n_embd = n_embd_val

    # CHANGED: Removed the suffix so we do NOT append run # anymore.
    if total_runs > 1:
        suffix = f"_run{run_idx+1}_of_{total_runs}"
        out_dir = out_dir + suffix

    per_token_flops = get_flop_per_token(n_embd, n_layer)
    total_flops = per_token_flops * D

    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        init_process_group(backend=backend)
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        used_device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(used_device)
        master_process = (ddp_rank == 0)
        seed_offset = ddp_rank
        assert gradient_accumulation_steps % ddp_world_size == 0
        actual_grad_steps = gradient_accumulation_steps // ddp_world_size
    else:
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
        used_device = device
        actual_grad_steps = gradient_accumulation_steps

    tokens_per_iter = actual_grad_steps * ddp_world_size * batch_size * block_size
    print(f"\n=== Starting run {run_idx+1}/{total_runs} with n_layer={n_layer}, n_embd={n_embd} ===")
    print(f"tokens per iteration: {tokens_per_iter}")
    if master_process:
        os.makedirs(out_dir, exist_ok=True)

    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device_type = 'cuda' if 'cuda' in used_device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # data
    data_dir = os.path.join('data', dataset)
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    if scaling == 'Kaplan' and scale_D:
        train_data = train_data[:D]
        print(f"Using {fraction_of_data} fraction of training data; #tokens: {D:.2e}")
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

    def get_batch(split):
        data_split = train_data if split == 'train' else val_data
        ix = torch.randint(len(data_split) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy(data_split[i:i+block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data_split[i+1:i+1+block_size].astype(np.int64)) for i in ix])
        if device_type == 'cuda':
            x, y = x.pin_memory().to(used_device, non_blocking=True), y.pin_memory().to(used_device, non_blocking=True)
        else:
            x, y = x.to(used_device), y.to(used_device)
        return x, y

    # vocab size from meta
    meta_path = os.path.join(data_dir, 'meta.pkl')
    meta_vocab_size = None
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        meta_vocab_size = meta['vocab_size']
        print(f"found vocab_size = {meta_vocab_size}")

    print(f"total flops: {total_flops:.2e}")

    # model init
    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                      block_size=block_size, bias=bias,
                      vocab_size=None, dropout=dropout)
    iter_num = 0
    best_val_loss = 1e9

    if init_from == 'scratch':
        print("Initializing a new model from scratch")
        if meta_vocab_size is None:
            print("defaulting to GPT-2 vocab_size=50304")
        model_args['vocab_size'] = meta_vocab_size if meta_vocab_size else 50304
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        print(f"max_iters: {max_iters}")
    elif init_from == 'resume':
        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
        print(f"Resuming from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=used_device)
        ckpt_model_args = checkpoint['model_args']
        for k in ['n_layer','n_head','n_embd','block_size','bias','vocab_size']:
            model_args[k] = ckpt_model_args[k]
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
    elif init_from.startswith('gpt2'):
        print(f"Initializing from GPT-2 weights: {init_from}")
        override_args = dict(dropout=dropout)
        model = GPT.from_pretrained(init_from, override_args)
        for k in ['n_layer','n_head','n_embd','block_size','bias','vocab_size']:
            model_args[k] = getattr(model.config, k)

    if block_size < model_args['block_size']:
        model.crop_block_size(block_size)
        model_args['block_size'] = block_size

    model.to(used_device)
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    if init_from == 'resume':
        optimizer.load_state_dict(checkpoint['optimizer'])

    if compile:
        print("compiling model, may take ~1 minute...")
        model = torch.compile(model)

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for sp in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                Xb, Yb = get_batch(sp)
                with ctx:
                    _, loss_ = model(Xb, Yb)
                losses[k] = loss_.item()
            out[sp] = losses.mean()
        model.train()
        return out

    def get_lr(step_i):
        if step_i < warmup_iters:
            return learning_rate * step_i / warmup_iters
        if step_i > lr_decay_iters:
            return min_lr
        decay_ratio = (step_i - warmup_iters) / (lr_decay_iters - warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (learning_rate - min_lr)

    # CHANGED: Retrieve number of params from model.py
    num_parameters = model.get_num_params(non_embedding=True)

    # CHANGED: Since we removed suffix, wandb name can still have run info:
    if wandb_log and master_process:
        import wandb
        wandb_name = f"{n_layer_val}l-{n_embd_val}e-{n_head}a-{num_parameters}p-r{run_idx+1}/{total_runs}"
        wandb.init(project=wandb_project, name=wandb_name, config=config, reinit=True)

    Xb, Yb = get_batch('train')
    t0 = time.time()
    local_iter_num = 0
    raw_model = model.module if ddp else model
    running_mfu = -1.0
    train_loss_records = []

    while True:
        lr_ = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_

        if iter_num % eval_interval == 0 and master_process and not (estimate_B_crit or scale_D == 'Chinchilla'):
            losses = estimate_loss()
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if wandb_log:
                import wandb
                wandb.log({
                    "iter": iter_num,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "lr": lr_,
                    "mfu": running_mfu*100,
                })
            if losses['val'] < best_val_loss or always_save_checkpoint:
                best_val_loss = losses['val']
                if iter_num > 0:
                    ckp = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': config,
                    }
                    print(f"saving checkpoint to {out_dir}")
                    torch.save(ckp, os.path.join(out_dir, 'ckpt.pt'))

        if iter_num == 0 and eval_only:
            break

        for micro_step in range(actual_grad_steps):
            if ddp:
                model.require_backward_grad_sync = (micro_step == actual_grad_steps - 1)
            with ctx:
                logits, loss_ = model(Xb, Yb)
                loss_ = loss_ / actual_grad_steps
            Xb, Yb = get_batch('train')
            scaler.scale(loss_).backward()

        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % log_interval == 0 and master_process:
            lossf = loss_.item() * actual_grad_steps
            if local_iter_num >= 5:
                mfu = raw_model.estimate_mfu(batch_size * actual_grad_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
            train_loss_records.append((iter_num, lossf, total_flops))

        iter_num += 1
        local_iter_num += 1
        if iter_num > max_iters:
            break

    if ddp:
        destroy_process_group()

    if wandb_log and master_process:
        import wandb
        wandb.finish()

    # Save final training curve to CSV
    if master_process:
        df_train = pd.DataFrame(train_loss_records, columns=["step","loss","flops"])
        df_train.to_csv(os.path.join(out_dir, "train_loss_records.csv"), index=False)

# multiple runs logic
if isinstance(n_layer, list) or isinstance(n_embd, list):
    if isinstance(n_layer, list):
        layer_list = n_layer
    else:
        layer_list = [n_layer]
    if isinstance(n_embd, list):
        embd_list = n_embd
    else:
        embd_list = [n_embd]

    combos = []
    for ln in layer_list:
        for em in embd_list:
            combos.append((ln, em))

    # uncomment if starting from specific run number
    for i, (ln_val, em_val) in enumerate(combos):
        if i < 20:
            continue
        run_single_train(ln_val, em_val, run_idx=i, total_runs=len(combos))

else:
    run_single_train(n_layer, n_embd, run_idx=0, total_runs=1)
