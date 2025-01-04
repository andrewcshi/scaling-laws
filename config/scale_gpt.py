wandb_log = True
wandb_project = 'owt-scaling'
wandb_run_id = "" # give only when resuming a W&B run
always_save_checkpoint = False 

# setting default values of scale_N, scale_D to False, you must change them from command line when scaling.
scaling = "Kaplan"
scale_N = False
scale_D = False

# replace n_layer, n_embd and fraction_of_data from command line. Default values:
n_layer = 4
n_embd = 64
fraction_of_data = 1.0

# Can also set n_head from command line, but set default value through this rule of thumb 
# given in Appendix F of 2010.14701
# It is consistent with nanoGPT where n_embd = 768, and n_head = 768//64 = 12.
n_head = 4

#### TRAINING CONFIGURATIONS FROM KAPLAN ET AL

# total batch size = 512 so set local batch size = 16, gradaccum = 32 
batch_size = 32
block_size = 256
gradient_accumulation_steps = 4 * 8

# total number of training iterations = 2.5e5
# learning rate warms up for 3000 iterations and decays to 0 at the end of training.
# dropout = 0.1 (see Section 4.2). minimum learning rate is 0
# maximum learning rate is given by equation D.1 of the paper. It depends on N, so we set it in configurator.py
max_iters = 6000
warmup_iters = 500
lr_decay_iters = 6000
dropout = 0.1
min_lr = 0

# eval stuff same as nanoGPT
eval_interval = 50
eval_iters = 200
log_interval = 1

# weight decay same as nanoGPT
weight_decay = 1e-1