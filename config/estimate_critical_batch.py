# config for estimating critical batch size as done in 1812.06162 and 2001.08361

wandb_log = True
wandb_project = 'owt-scaling'
wandb_run_id = "" # give only when resuming a W&B run
always_save_checkpoint = False 

# technically, estimating critical batch size is independent of scaling, but setting this to 'Kaplan' as we perform critical batch computation to use it in Kaplan scaling laws.
scaling = 'Kaplan' 
estimate_B_crit = True

# set n_layer, n_embd, and n_head here. 
n_layer = 8
n_embd = 128
n_head = max(2, n_embd // 64) # same as in scale_gpt. 

# will be sweeping over learning rate and batch size * gradient accumulation steps
# but let the following values be default
learning_rate = 1.2e-3
batch_size = 16
gradient_accumulation_steps = 32

#### TRAINING CONFIGURATIONS FROM KAPLAN ET AL

block_size = 1024

# total number of training iterations = 2.5e5
# learning rate warms up for 3000 iterations and decays to 0 at the end of training.
# dropout = 0.1 (see Section 4.2). minimum learning rate is 0
# maximum learning rate is given by equation D.1 of the paper. It depends on N, so we set it in configurator.py
max_iters = int(2.5e5)
warmup_iters = 3000
lr_decay_iters = int(2.5e5)
dropout = 0.1 
min_lr = 0 

# eval stuff is different when measuring critical batch
# Following 1812.06162, we keep track of training loss and smooth it later
# so set eval_interval to a larger number like 2*max_iters
eval_interval = 2*max_iters
log_interval = 5 # but log training loss a bit more frequently

# weight decay same as nanoGPT
weight_decay = 1e-1