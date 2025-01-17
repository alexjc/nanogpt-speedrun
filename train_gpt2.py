import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import uuid
import time
import glob
import subprocess
import contextlib
from dataclasses import dataclass
from pathlib import Path

import torch
import torch._inductor.config as config
torch.empty(1, device='cuda', requires_grad=True).backward() # What does this do?
from torch import Tensor, nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
from torch.nn.parallel import DistributedDataParallel as DDP
# use of FlexAttention contributed by @KoszarskyB
from torch.nn.attention.flex_attention import BlockMask, flex_attention

config.coordinate_descent_tuning = True
flex_kernel_options = None
if torch.cuda.get_device_name(0).endswith(("3090", "4090")):
    flex_kernel_options = {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_M1": 32, "BLOCK_N1": 64, "BLOCK_M2": 64, "BLOCK_N2": 32}

torch.manual_seed(12345)
torch.cuda.manual_seed_all(12345)

# -----------------------------------------------------------------------------
# Muon optimizer

@torch.compile
def zeropower_via_newtonschulz5(G, steps):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T

    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X
    
    if G.size(0) > G.size(1):
        X = X.T
    return X

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer assumes that all parameters passed in are 2D.
    - It should not be used for the embedding layer, the final fully connected layer, or any {0,1}-D
    parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.
    - We believe it is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.
    - We have not yet tried this optimizer for training scenarios larger than NanoGPT (124M).

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iteration steps to use.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        self.world_size = int(os.environ['WORLD_SIZE'])
        self.rank = int(os.environ['RANK'])
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        assert all(isinstance(p, torch.Tensor) for p in params)
        sizes = {p.numel() for p in params}
        param_groups = [dict(params=[p for p in params if p.numel() == size],
                             update_buffer=[torch.empty(size, device='cuda', dtype=torch.bfloat16) for _ in range(self.world_size)])
                        for size in sizes]
        super().__init__(param_groups, defaults)
        # pre-init momentum
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['momentum_buffer'] = torch.zeros(p.shape, device=p.device, dtype=torch.float)

    def step(self):

        for group in self.param_groups:

            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']
            update_buffers = group['update_buffer']
            # generate weight updates in distributed fashion
            params = group['params']
            handle = None
            params_world = None
            def update_prev():
                if params_world is None:
                    return
                assert handle is not None
                handle.wait()
                for p_world, g_world in zip(params_world, update_buffers):
                    p_world.data.add_(
                        g_world.view_as(p_world),
                        alpha=-lr * max(1, p_world.size(0) / p_world.size(1)) ** 0.5,
                    )
            for base_i in range(len(params))[::self.world_size]:
                if base_i + rank < len(params):
                    p = params[base_i + self.rank]
                    g = p.grad
                    assert g is not None
                    state = self.state[p]

                    buf = state['momentum_buffer']
                    buf.lerp_(g, 1 - momentum)
                    g = g.lerp_(buf, momentum) if nesterov else buf
                    g = zeropower_via_newtonschulz5(g, steps=ns_steps).flatten()
                else:
                    g = update_buffers[rank]
                update_prev() # async all_gather instead of sync all_reduce by @YouJiacheng
                handle = dist.all_gather(update_buffers, g, async_op=True)
                params_world = params[base_i : base_i + self.world_size]
            update_prev()

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the GPT-2 model

# semi-orthogonal initialization @fernbear.bsky.social
def semi_orthogonal_init(dim_in, dim_out, steps=5):
    # default pytorch linear layer init
    weight= (1/dim_in)**.5*(2*(torch.rand(dim_out, dim_in, device='cuda')) - 1.)
    
    w_std = weight.std(dim=0).mean()
    weight = zeropower_via_newtonschulz5(weight, steps=steps)
    weight *= w_std/weight.std(dim=0).mean().add_(1e-8)
    return weight.float()
    
# fused scalar <-> RMSNorm @fernbear.bsky.social
class RMSNormScalar(nn.Module):
    def __init__(self, init_val=.5413):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(init_val))  # Softplus of default value is ~1

    def forward(self, x):
        norm = x.float().norm(dim=-1, keepdim=True).type_as(x)
        norm = norm.add(torch.finfo(norm.dtype).eps)
        rms_inv = (F.softplus(self.scale.type_as(x)) * x.shape[-1] ** .5) / norm
        result = x * rms_inv
        return result

class CastedLinear(nn.Linear):

    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=False)

    def forward(self, x):
        return F.linear(x, self.weight.type_as(x))

class Rotary(nn.Module):

    def __init__(self, dim, max_seq_len=65536):
        super().__init__()
        # half-truncate RoPE by @YouJiacheng
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim//4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum('i,j -> ij', t, angular_freq)
        self.cos = nn.Buffer(theta.cos(), persistent=False)
        self.sin = nn.Buffer(theta.sin(), persistent=False)

    def forward(self, x):
        cos, sin = self.cos[None, :x.size(-3), None, :], self.sin[None, :x.size(-3), None, :]
        x1, x2 = x.float().chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x)

class CausalSelfAttention(nn.Module):

    def __init__(self, dim, num_heads, v_res=True):
        super().__init__()
        assert dim % num_heads == 0
        num_heads = num_heads
        self.num_heads = num_heads
        self.c_q = nn.Parameter(semi_orthogonal_init(dim, dim))
        self.c_k = nn.Parameter(semi_orthogonal_init(dim, dim))
        self.c_v = nn.Parameter(semi_orthogonal_init(dim, dim))
        self.qkv_scale = nn.Parameter(torch.ones(3*dim))
        self.q_norm = RMSNormScalar()
        self.k_norm = RMSNormScalar()
        if v_res:
            self.v_lambda = nn.Parameter(torch.tensor([-.4328])) #~zero init after softplus
        self.rotary = Rotary(dim // num_heads)
        self.c_proj = nn.Parameter(torch.zeros(dim, dim))
        self.c_proj_scale = nn.Parameter(torch.ones(dim))

    def forward(self, x, ve, block_mask):
        B, T = x.size(0), x.size(1) # batch size, sequence length
        assert B == 1, 'Must use batch size = 1 for FlexAttention'
        qkv_weight = (self.qkv_scale.unsqueeze(1) * torch.cat([self.c_q, self.c_k, self.c_v], dim=0)).type_as(x) # fuse weights @fernbear.bsky.social
        q, k, v = F.linear(x, qkv_weight).view(B, T, 3*self.num_heads, -1).chunk(3, dim=-2)
        if ve is not None: # skip mid-layers token value embeddings by @YouJiacheng
            v = v + F.softplus(self.v_lambda.type_as(v)) * ve.view_as(v) # @KoszarskyB & @Grad62304977 & @fernbear.bsky.social
        q, k = self.q_norm(q), self.k_norm(k) # QK norm @Grad62304977, learnable scalar @brendanh0gan
        q, k = self.rotary(q), self.rotary(k)
        y = flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=block_mask, kernel_options=flex_kernel_options)
        y = y.transpose(1, 2).contiguous().view_as(x) # re-assemble all head outputs side by side
        y = F.linear(y, (self.c_proj_scale*self.c_proj).type_as(x))
        return y

class MLP(nn.Module):
    def __init__(self, dim, expand=4):
        super().__init__()
        expand_dim = expand*dim
        self.c_fc_scale   = nn.Parameter(torch.ones(dim))
        self.c_fc         = nn.Parameter(semi_orthogonal_init(dim, expand_dim))
        self.c_proj       = nn.Parameter(torch.zeros(dim, expand_dim))
        self.c_proj_scale = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        x = F.linear(x, (self.c_fc*self.c_fc_scale.unsqueeze(0)).type_as(x)) # fuse weight & weight_scale mults
        x = F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = F.linear(x, (self.c_proj_scale.unsqueeze(1) * self.c_proj).type_as(x))
        return x

class Block(nn.Module):

    def __init__(self, model_dim, num_heads, block_num, use_attn=True):
        super().__init__()
        self.attn = CausalSelfAttention(model_dim, num_heads, v_res=(block_num in [0, 1, 2, 9, 10, 11])) if use_attn else None
        self.attn_norm = RMSNormScalar(-.4328) if self.attn is not None else None
        self.mlp = MLP(model_dim)
        self.mlp_norm  = RMSNormScalar()
        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))

    def forward(self, x, ve, x0, block_mask):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        if self.attn is not None:
            x = x + self.attn(self.attn_norm(x), ve, block_mask)
        x = x + self.mlp(self.mlp_norm(x))
        return x

class ValueEmbedding(nn.Module):
    def __init__(self, vocab_size, model_dim):
        super().__init__()
        self.embed = nn.ModuleList([nn.Embedding(vocab_size, model_dim).bfloat16() for _ in range(3)])

    def forward(self, inputs):
        ve = [emb(inputs) for emb in self.embed]  
        # 012 ... 012 structure on token value embeddings by @YouJiacheng, improved on @leloykun's U-net structure
        ve = [ve[0], ve[1], ve[2], None, None, None, None, None, None, ve[0], ve[1], ve[2]]
        return ve

# -----------------------------------------------------------------------------
# The main GPT-2 model

class GPT(nn.Module):

    def __init__(self, vocab_size, num_layers, num_heads, model_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim)
        self.embed_norm = RMSNormScalar()
        # skip attention of blocks.7 (the 8th layer) by @YouJiacheng
        self.blocks = nn.ModuleList([Block(model_dim, num_heads, i, use_attn=(i != 7))
                                     for i in range(num_layers)])
        # token value embeddings by @KoszarskyB - inspired by @Grad62304977's value residual learning
        # U-net structure on token value embeddings by @leloykun
        self.value_embeds = ValueEmbedding(vocab_size, model_dim)
        self.lm_head = nn.Parameter(torch.zeros(vocab_size, model_dim)) # zero init - @Grad6230497
        self.lm_head_scale = nn.Parameter(torch.ones(model_dim))
        self.out_norm = RMSNormScalar()
        self.out_bias = nn.Embedding(vocab_size, model_dim)
        # U-net design by @brendanh0gan
        self.num_encoder_layers = num_layers // 2 # Half of the layers for encoder
        self.num_decoder_layers = num_layers - self.num_encoder_layers # Remaining for decoder
        # Add learnable skip connection weights for decoder layers
        self.skip_weights = nn.Parameter(torch.ones(self.num_decoder_layers))
        self.register_buffer('noise_scale', torch.tensor(0.1))

    def forward(self, inputs, targets, sliding_window_num_blocks):
        BLOCK_SIZE = 128
        seq_len = len(inputs)
        assert seq_len % BLOCK_SIZE == 0
        total_num_blocks = seq_len // BLOCK_SIZE
        assert inputs.ndim == 1
        docs = (inputs == 28415).cumsum(0)
        docs_low = docs.view(-1, BLOCK_SIZE)[:, 0].contiguous()
        docs_high = docs.view(-1, BLOCK_SIZE)[:, -1].contiguous()

        def document_causal(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            document_mask = docs[q_idx] == docs[kv_idx]
            return causal_mask & document_mask

        def dense_to_ordered(dense_mask):
            num_blocks = dense_mask.sum(dim=-1, dtype=torch.int32)
            indices = dense_mask.argsort(dim=-1, descending=True, stable=True).to(torch.int32)
            return num_blocks[None, None].contiguous(), indices[None, None].contiguous()

        def create_doc_swc_block_mask(sliding_window_num_blocks):
            kv_idx = block_idx = torch.arange(total_num_blocks, dtype=torch.int32, device='cuda')
            q_idx = block_idx[:, None]
            causal_bm = q_idx >= kv_idx
            causal_full_bm = q_idx > kv_idx
            window_bm = q_idx - kv_idx < sliding_window_num_blocks
            window_full_bm = window_bm # block-wise sliding window by @YouJiacheng
            # document_bm = (docs_low[q_idx] <= docs_high[kv_idx]) & (docs_low[kv_idx] <= docs_high[q_idx])
            document_bm = (docs_low[:, None] <= docs_high) & (docs_low <= docs_high[:, None])
            document_full_bm = (docs_low[:, None] == docs_high) & (docs_low == docs_high[:, None])
            nonzero_bm = causal_bm & window_bm & document_bm
            full_bm  = causal_full_bm & window_full_bm & document_full_bm
            kv_num_blocks, kv_indices = dense_to_ordered(nonzero_bm & ~full_bm)
            full_kv_num_blocks, full_kv_indices = dense_to_ordered(full_bm)
            return BlockMask.from_kv_blocks(
                kv_num_blocks,
                kv_indices,
                full_kv_num_blocks,
                full_kv_indices,
                BLOCK_SIZE=BLOCK_SIZE,
                mask_mod=document_causal,
            )

        block_mask = create_doc_swc_block_mask(sliding_window_num_blocks)

        embeds = self.embed(inputs[None])
        noise = self.noise_scale * torch.fmod(torch.randn_like(embeds), 2.) # input noising by schedule by @brendanh0gan
        x0 = self.embed_norm(embeds + noise).bfloat16() # use of norm here by @Grad62304977, scalar norm by @fernbear.bsky.social
        x = x0
        ve = self.value_embeds(inputs) 
        assert len(ve) == len(self.blocks)
        ve_enc, ve_dec = ve[:self.num_encoder_layers], ve[self.num_encoder_layers:]

        # Store outputs for U-Net skip connections
        skip_connections = []
        # Encoder pass - process only the first half of the blocks
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, ve_enc[i], x0, block_mask)
            skip_connections.append(x)
        # Decoder pass - process the remaining blocks with weighted skip connections
        for i in range(self.num_decoder_layers):
            x = x + self.skip_weights[i] * skip_connections.pop()
            # U-net structure on token value embeddings by @leloykun
            x = self.blocks[self.num_encoder_layers + i](x, ve_dec[i], x0, block_mask)

        x = self.out_norm(x - x0 + self.out_bias(inputs).bfloat16()) # input subtract, per-token output bias by @fernbear.bsky.social
        logits = F.linear(x, ((self.lm_head_scale / 7.5) * self.lm_head).type_as(x)) # fuse sigmoid scale & weight scale @fernbear.bsky.social
        logits = 30 * torch.sigmoid(logits) # @Grad62304977 added tanh softcapping, @KoszarskyB reduced it from 30 to 15, @YuJiacheng tanh -> sigmoid + different scaling
        logits = logits.float()
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets)
        return loss


# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

def _load_data_shard(path):
    # only reads the header, returns header data
    # header is 256 int32
    header = torch.from_file(path, False, 256, dtype=torch.int32)
    assert header[0] == 20240520, 'magic number mismatch in the data .bin file'
    assert header[1] == 1, 'unsupported version'
    num_tokens = int(header[2]) # number of tokens (claimed)
    with open(path, 'rb', buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True) # avoid pin_memory copy by @YouJiacheng
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy by @YouJiacheng
        assert nbytes == 2 * num_tokens, 'number of tokens read does not match header'
    return tokens

class DistributedDataLoader:

    def __init__(self, filename_pattern):
        self.rank = int(os.environ['RANK'])
        self.world_size = int(os.environ['WORLD_SIZE'])
        self.files = sorted(glob.glob(filename_pattern))
        self.reset()

    def reset(self):
        self.current_shard = -1
        self.advance()

    def advance(self):
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = 0
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self, batch_size):
        assert batch_size % self.world_size == 0
        device_batch_size = batch_size // self.world_size
        # load next shard if necessary
        if self.current_position + batch_size + 1 >= len(self.tokens):
            self.advance()
        pos = self.current_position + self.rank * device_batch_size
        device_batch_tokens = self.tokens[pos:pos+device_batch_size+1]
        # advance current position
        self.current_position += batch_size
        inputs = device_batch_tokens[:-1].to(device='cuda', dtype=torch.int32, non_blocking=True)
        targets = device_batch_tokens[1:].to(device='cuda', dtype=torch.int64, non_blocking=True)
        return inputs, targets


# -----------------------------------------------------------------------------
# int main

@dataclass
class Hyperparameters:
    # data
    train_bin = 'data/fineweb-tokmon-10B/fineweb-tokmon_train_v2_*.bin' # input .bin to train on
    val_bin = 'data/fineweb-tokmon-10B/fineweb-tokmon_val_v2_*.bin' # input .bin to eval validation loss on
    # optimization
    batch_size = 8*64*1024 # batch size in tokens
    max_device_batch_size = 32*1024 # batch size per device in tokens
    num_iterations = 1330 #1350 #1390 # number of iterations to run
    cooldown_frac = 0.4 # fraction of training spent cooling down the learning rate
    bf16_embeds = True
    # evaluation and logging
    val_loss_every = 125 # every how many steps to evaluate val loss? 0 for only at the end
    # val_tokens = 10485760 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    val_tokens = 10420224 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    val_ratio = 0.99011
    # implementation
    save_checkpoint = False
args = Hyperparameters()

micro_bs = args.max_device_batch_size

# set up DDP (distributed data parallel). torchrun sets this env variable
rank = int(os.environ['RANK'])
local_rank = int(os.environ['LOCAL_RANK'])
world_size = int(os.environ['WORLD_SIZE'])
assert torch.cuda.is_available()
torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl', device_id=torch.device(local_rank))
dist.barrier()
master_process = (rank == 0) # this process will do logging, checkpointing etc.

# begin logging
logfile = None
if master_process:
    run_id = uuid.uuid4()
    os.makedirs('logs', exist_ok=True)
    logfile = f'logs/{run_id}.txt'
    print(logfile)

def print0(s, console=False):
    if master_process:
        with open(logfile, 'a') as f:
            if console:
                print(s)
            print(s, file=f)

# begin by printing this file (the Python code)
print0(code)
print0('='*100)
# log information about the hardware/software environment this is running on
print0(f'Running Python {sys.version}')
print0(f'Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}')
print0(subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout)
print0('='*100)

# load data
train_loader = DistributedDataLoader(args.train_bin)
val_loader = DistributedDataLoader(args.val_bin)
print0(f'Training dataloader files: {train_loader.files}')
print0(f'Validation dataloader files: {val_loader.files}')
print0('='*100)

# there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency. suggested to me by @Grad62304977.
# this originates from Karpathy's experiments.
model = GPT(vocab_size=28416, num_layers=12, num_heads=6, model_dim=768)
model = model.cuda()
if args.bf16_embeds:
    for m in model.modules():
        if isinstance(m, nn.Embedding):
            m.bfloat16()

model = torch.compile(model)
ddp_model = DDP(model, device_ids=[local_rank], broadcast_buffers=False, gradient_as_bucket_view=True)

# collect the parameters to optimize
hidden_matrix_params = [p for p in model.blocks.parameters() if p.ndim == 2]
embed_params = [model.out_bias.weight, model.embed.weight, *model.value_embeds.parameters()]
scalar_params = [p for p in model.parameters() if p.ndim < 2]
head_params = [model.lm_head]


# init the optimizer(s)
k = 1.06
optimizer1 = torch.optim.Adam([dict(params=embed_params, lr=0.6*k),
                              dict(params=head_params, lr=0.008*k),
                              dict(params=scalar_params, lr=0.04*k)],
                              betas=(0.8, 0.95), fused=True)
optimizer2 = Muon(hidden_matrix_params, lr=0.05*k, momentum=0.95)
optimizers = [optimizer1, optimizer2]

# learning rate schedule: stable then decay
def get_lr(it):
    t = 1 - it / args.num_iterations # time remaining in training
    assert 1 >= t >= 0
    w = min(t / args.cooldown_frac, 1.0) # 1 -> 0
    return w * 1.0 + (1 - w) * 0.1

schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]

# sliding window size schedule: linear increase over training in chunks of 128 from 128 -> 1792. By @fernbear.bsky.social
def get_sliding_window_blocks(it):
    x = it / args.num_iterations # training progress
    x = max(min(x, 1.), 0.)
    return int(((1 - x) * 128 + x * 1856) // 128)
sliding_window_num_blocks = torch.tensor(1, dtype=torch.int32, device='cuda')

# Start training loop
training_time_ms = 0
# start the clock
torch.cuda.synchronize()
t0 = time.perf_counter()
# begin training
train_steps = args.num_iterations
for step in range(train_steps + 1):
    last_step = (step == train_steps)
    # This effectively ignores timing first 10 steps, which are slower for weird reasons.
    # Alternately, and slightly more correctly in terms of benchmarking, we could do 10
    # steps with dummy data first, and then re-initialize the model and reset the loader.
    if step == 10:
        training_time_ms = 0
        t0 = time.perf_counter()
    timed_steps = float('nan') if step <= 11 else (step - 10) + 1 # <= 11 to avoid bug in val

    sliding_window_num_blocks.copy_(get_sliding_window_blocks(step), non_blocking=True)

    # Update noise scale to follow triangular schedule
    # Cosine decay from 0.01 to 0 over entire training
    progress = step / args.num_iterations 
    noise_scale = torch.cos(torch.tensor(progress * torch.pi)).mul_(.01)
    model.noise_scale.copy_(noise_scale, non_blocking=True) # update model with new noise scale

    # --------------- VALIDATION SECTION -----------------
    if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.perf_counter() - t0)
        # run validation batches
        model.eval()
        val_loader.reset()
        val_loss = 0.0

        # copy the noise scale out temporarily
        noise_scale_cache = model.noise_scale.clone()
        model.noise_scale.zero_()

        # calculate the number of steps to take in the val loop.
        val_batch_size = world_size * micro_bs
        assert args.val_tokens % val_batch_size == 0
        val_steps = args.val_tokens // val_batch_size
        for _ in range(val_steps):
            with torch.no_grad():
                inputs_val, targets_val = val_loader.next_batch(val_batch_size)
                val_loss += ddp_model(inputs_val, targets_val, sliding_window_num_blocks)
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        val_loss = (val_loss * args.val_ratio) / val_steps
        # logging
        print0(f'step:{step}/{train_steps} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms', console=True)
        # reset noise value
        model.noise_scale.copy_(noise_scale_cache, non_blocking=True)
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.perf_counter()

    if last_step:
        if master_process and args.save_checkpoint:
            log = dict(step=step, code=code, model=model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
            os.makedirs(f'logs/{run_id}', exist_ok=True)
            torch.save(log, f'logs/{run_id}/state_step{step:06d}.pt')
        # the last step only has the validation loop, so break to avoid training
        break

    # --------------- TRAINING SECTION -----------------
    model.train()
    batch_size = args.batch_size
    assert batch_size % world_size == 0
    inputs_train, targets_train = train_loader.next_batch(batch_size)
    assert len(inputs_train) <= micro_bs or len(inputs_train) % micro_bs == 0
    for micro_inputs_train, micro_targets_train in zip(inputs_train.split(micro_bs), targets_train.split(micro_bs)):
        ddp_model(micro_inputs_train, micro_targets_train, sliding_window_num_blocks).mul(world_size).backward()
    # momentum warmup for Muon
    frac = min(step/300, 1)
    for group in optimizer2.param_groups:
        group['momentum'] = (1 - frac) * 0.85 + frac * 0.95
    # step the optimizers and schedulers
    for opt, sched in zip(optimizers, schedulers):
        opt.step()
        if step != train_steps-1:
            sched.step()

    # null the gradients
    model.zero_grad(set_to_none=True)
   
    # logging
    approx_time = training_time_ms + 1000 * (time.perf_counter() - t0)
    print0(f'step:{step+1}/{train_steps} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms', console=True)

print0(f'peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB')
dist.destroy_process_group()