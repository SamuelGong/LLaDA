"""qcache_baseline.py – vanilla vs. naive Q‑cache benchmark

Adds **output text printing (outside timing)** so you can eyeball whether the two
modes generate identical/acceptable answers. Still keeps model‑load outside the
measured section and avoids double‑loading into memory.

Run example:
    python qcache_baseline.py --question "What is diffusion?" --steps 128 --gen 128
"""

import argparse
from contextlib import contextmanager
import torch
from transformers import AutoTokenizer, AutoModel
from generate import generate

import types

# ───────────────────────────────────────── config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16
MODEL_NAME = "GSAI-ML/LLaDA-8B-Instruct"

# ─────────────────────────────────── timing helper
@contextmanager
def cuda_timer(label="Elapsed"):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    yield lambda: start.elapsed_time(end) / 1000  # returns seconds
    end.record(); torch.cuda.synchronize()
    print(f"{label}: {start.elapsed_time(end)/1000:.3f}s")

# ───────────────────────────── generate with callback (same as before)
# @torch.no_grad()
# def generate_with_callback(model, prompt, *, steps=128, gen_length=128, block_length=128,
#                            temperature=0., cfg_scale=0., remasking='low_confidence',
#                            mask_id=126336, step_callback=None):
#     x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=model.device)
#     x[:, :prompt.shape[1]] = prompt.clone()
#     prompt_idx = (x != mask_id)
#
#     assert gen_length % block_length == 0
#     n_blocks = gen_length // block_length
#     assert steps % n_blocks == 0
#     s_per_block = steps // n_blocks
#
#     for b in range(n_blocks):
#         blk_slice = slice(prompt.shape[1] + b * block_length, prompt.shape[1] + (b + 1) * block_length)
#         blk_mask = (x[:, blk_slice] == mask_id)
#         n_trans = get_num_transfer_tokens(blk_mask, s_per_block)
#
#         for t in range(s_per_block):
#             mask_idx = (x == mask_id)
#             if cfg_scale > 0.:
#                 un_x = x.clone(); un_x[prompt_idx] = mask_id
#                 logits, un_logits = model(torch.cat([x, un_x], 0)).logits.chunk(2, 0)
#                 logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
#             else:
#                 logits = model(x).logits
#
#             logits = add_gumbel_noise(logits, temperature)
#             x0 = logits.argmax(-1)
#
#             if remasking == 'low_confidence':
#                 p = torch.softmax(logits, -1)
#                 x0_p = p.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
#             elif remasking == 'random':
#                 x0_p = torch.rand_like(x0, dtype=logits.dtype)
#             else:
#                 raise NotImplementedError
#
#             x0_p[:, prompt.shape[1] + (b + 1) * block_length:] = -float('inf')
#             x0 = torch.where(mask_idx, x0, x)
#             conf = torch.where(mask_idx, x0_p, -float('inf'))
#             transfer = torch.zeros_like(x, dtype=torch.bool)
#             for j in range(conf.size(0)):
#                 _, idx = torch.topk(conf[j], k=n_trans[j, t])
#                 transfer[j, idx] = True
#             x[transfer] = x0[transfer]
#
            # if step_callback is not None:
            #     step_callback(b * s_per_block + t, (x == mask_id))
#     return x

# @ torch.no_grad()
# def generate_with_callback(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
#              cfg_scale=0., remasking='low_confidence', mask_id=126336, step_callback=None):
#     '''
#     Args:
#         model: Mask predictor.
#         prompt: A tensor of shape (1, L).
#         steps: Sampling steps, less than or equal to gen_length.
#         gen_length: Generated answer length.
#         block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
#         temperature: Categorical distribution sampling temperature.
#         cfg_scale: Unsupervised classifier-free guidance scale.
#         remasking: Remasking strategy. 'low_confidence' or 'random'.
#         mask_id: The toke id of [MASK] is 126336.
#     '''
#     x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
#     x[:, :prompt.shape[1]] = prompt.clone()
#
#     prompt_index = (x != mask_id)
#
#     assert gen_length % block_length == 0
#     num_blocks = gen_length // block_length
#
#     assert steps % num_blocks == 0
#     steps = steps // num_blocks
#
#     for num_block in range(num_blocks):
#         block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
#         num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
#         for i in range(steps):
#             mask_index = (x == mask_id)
#             if cfg_scale > 0.:
#                 un_x = x.clone()
#                 un_x[prompt_index] = mask_id
#                 x_ = torch.cat([x, un_x], dim=0)
#                 logits = model(x_).logits
#                 logits, un_logits = torch.chunk(logits, 2, dim=0)
#                 logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
#             else:
#                 logits = model(x).logits
#
#             logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
#             x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
#
#             if remasking == 'low_confidence':
#                 p = F.softmax(logits, dim=-1)
#                 x0_p = torch.squeeze(
#                     torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
#             elif remasking == 'random':
#                 x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
#             else:
#                 raise NotImplementedError(remasking)
#
#             x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf
#
#             x0 = torch.where(mask_index, x0, x)
#             confidence = torch.where(mask_index, x0_p, -np.inf)
#
#             transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
#             for j in range(confidence.shape[0]):
#                 _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
#                 transfer_index[j, select_index] = True
#             x[transfer_index] = x0[transfer_index]
#
#             if step_callback is not None:
#                 step_callback(num_block * steps + i, (x == mask_id))
#
#     return x

# ─────────────────────────────── find transformer blocks

def find_blocks(model):
    for path in [("model", "transformer", "blocks"), ("model", "transformer", "layers"), ("model", "layers"), ("model", "h")]:
        obj = model
        for attr in path:
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                break
        else:
            return obj
    raise ValueError("Cannot locate transformer blocks in model")

# ───────────────────────────────── attach Q‑cache (unchanged from last version)

# def attach_qcache(model, total_seq_len):
#     blocks = find_blocks(model)
#     n_layers = len(blocks)
#     d_model = model.config.hidden_size
#
#     q_mem = torch.empty(n_layers, total_seq_len, d_model, dtype=DTYPE, device=DEVICE)
#     valid = torch.zeros(n_layers, total_seq_len, dtype=torch.bool, device=DEVICE)
#
#     def make_hooks(lidx, fused):
#         def pre_hook(module, input):
#             if fused:
#                 return None
#             if valid[lidx].all():
#                 return (q_mem[lidx:lidx + 1],)
#             return None
#
#         def post_hook(module, input, output):
#             if fused:
#                 d = output.size(-1) // 3
#                 q, kv = output[..., :d], output[..., d:]
#             else:
#                 q = output
#             mixed_q = torch.where(valid[lidx, :, None], q_mem[lidx], q)
#             q_mem[lidx] = torch.where(valid[lidx, :, None], q_mem[lidx], q)
#             valid[lidx] |= True
#             return torch.cat([mixed_q, kv], -1) if fused else mixed_q
#         return pre_hook, post_hook
#
#     for lidx, blk in enumerate(blocks):
#         if hasattr(blk, "q_proj"):
#             pre, post = make_hooks(lidx, fused=False)
#             blk.q_proj.register_forward_pre_hook(pre)
#             blk.q_proj.register_forward_hook(post)
#         elif hasattr(blk, "att_proj"):
#             _, post = make_hooks(lidx, fused=True)
#             blk.att_proj.register_forward_hook(post)
#
#     def step_reset(mask_tensor):
#         decoded = (~mask_tensor.bool())[0]
#         valid[:, decoded] = False
#
#     return step_reset


def attach_qcache_monkey(model, seq_len,
                         device="cuda", dtype=torch.bfloat16):
    blocks   = find_blocks(model)              # 你的辅助函数
    n_layers = len(blocks)
    d_model  = model.config.hidden_size

    # 每层一次性缓存 (seq_len, d_model) 的 Query
    q_mem  = torch.empty(n_layers, seq_len, d_model,
                         device=device, dtype=dtype)
    cached = [False] * n_layers                # layer-level flag

    # ------------ 把 q_proj.forward 打补丁 ------------------------
    def patch_linear(lidx: int, lin: torch.nn.Linear):
        orig_fwd = lin.forward                # 保存原 BoundMethod

        def new_forward(self, x, *args, **kwargs):
            # 若已缓存 ➜ 直接返回缓存张量，跳过 GEMM
            if cached[lidx]:
                # 扩 batch 维以适配 (B, L, d); 这里默认 B==1
                return q_mem[lidx:lidx+1].to(x.dtype)

            # 第一次调用 → 正常 GEMM
            out = orig_fwd(x, *args, **kwargs)    # (1, L, d_q)
            q_mem[lidx] = out[0].to(dtype)        # 仅支持 batch==1
            cached[lidx] = True
            return out

        # 用 types.MethodType 绑定到实例
        lin.forward = types.MethodType(new_forward, lin)

    # 只 patch 分离的 q_proj；模型里没有 att_proj，你已确认
    for lidx, blk in enumerate(blocks):
        if hasattr(blk, "q_proj"):
            patch_linear(lidx, blk.q_proj)


# ─────────────────────────────────── benchmark helper

def benchmark(prompt, tokenizer, *, steps, gen_len, block_len, use_qcache):
    tag = "Q‑cache" if use_qcache else "Vanilla"
    print(f"\nLoading model for {tag} …")
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True, torch_dtype=DTYPE).to(DEVICE).eval()

    # warm‑up
    with torch.inference_mode():
        _ = model(prompt[:, :1]); torch.cuda.synchronize()

    attach_qcache_monkey(model, prompt.shape[1] + gen_len) if use_qcache else None
    with cuda_timer(f"{tag}") as get_elapsed:
        out = generate(model, prompt, steps=steps, gen_length=gen_len,
                                     block_length=block_len, temperature=0., cfg_scale=0.,
                                     remasking='low_confidence')
    # decode and show (outside timing)
    answer = tokenizer.batch_decode(out[:, prompt.shape[1]:], skip_special_tokens=True)[0]
    print(f"{tag} output → {answer}\n")

    # free memory
    del model; torch.cuda.empty_cache()

# ─────────────────────────────────────── main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", default="Explain diffusion models briefly.")
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--block", type=int, default=32)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    prompt_txt = tokenizer.apply_chat_template([{"role": "user", "content": args.question}], add_generation_prompt=True, tokenize=False)
    prompt = torch.tensor(tokenizer(prompt_txt)["input_ids"], device=DEVICE).unsqueeze(0)

    benchmark(prompt, tokenizer, steps=args.steps, gen_len=args.gen, block_len=args.block, use_qcache=False)
    benchmark(prompt, tokenizer, steps=args.steps, gen_len=args.gen, block_len=args.block, use_qcache=True)

if __name__ == "__main__":
    main()
