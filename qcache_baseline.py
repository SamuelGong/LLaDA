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

from generate import generate, generate_with_callback
import types
from torch.profiler import profile, ProfilerActivity
import torch.nn.functional as F
import math

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


# def attach_qcache_monkey(model, seq_len,
#                          device="cuda", dtype=torch.bfloat16):
#     blocks   = find_blocks(model)              # 你的辅助函数
#     n_layers = len(blocks)
#     d_model  = model.config.hidden_size
#
#     # 每层一次性缓存 (seq_len, d_model) 的 Query
#     q_mem  = torch.empty(n_layers, seq_len, d_model,
#                          device=device, dtype=dtype)
#     cached = [False] * n_layers                # layer-level flag
#
#     # ------------ 把 q_proj.forward 打补丁 ------------------------
#     def patch_linear(lidx: int, lin: torch.nn.Linear):
#         orig_fwd = lin.forward                # 保存原 BoundMethod
#
#         def new_forward(self, x, *args, **kwargs):
#             # 若已缓存 ➜ 直接返回缓存张量，跳过 GEMM
#             if cached[lidx]:
#                 # 扩 batch 维以适配 (B, L, d); 这里默认 B==1
#                 return q_mem[lidx:lidx+1].to(x.dtype)
#
#             # 第一次调用 → 正常 GEMM
#             out = orig_fwd(x, *args, **kwargs)    # (1, L, d_q)
#             q_mem[lidx] = out[0].to(dtype)        # 仅支持 batch==1
#             cached[lidx] = True
#             return out
#
#         # 用 types.MethodType 绑定到实例
#         lin.forward = types.MethodType(new_forward, lin)
#
#     # 只 patch 分离的 q_proj；模型里没有 att_proj，你已确认
#     for lidx, blk in enumerate(blocks):
#         if hasattr(blk, "q_proj"):
#             patch_linear(lidx, blk.q_proj)


def attach_qkv_full_cache(model, seq_len,
                          device="cuda", dtype=torch.bfloat16):
    """
    • Q : 首步算，后续整层复用
    • K/V: 逐行稀疏重算 —— 只有在“刚解码”的位置重算一次
    • Attention: 只对新增列做 QKᵀ、softmax、AV
    返回 step_reset(mask) 供采样循环在每步调用
    """
    blocks     = find_blocks(model)                 # 你已有的工具
    n_layers   = len(blocks)
    d_model    = model.config.hidden_size
    n_heads    = model.config.num_attention_heads
    d_head     = d_model // n_heads

    # ---------------- 缓存区 ------------------------
    q_mem = torch.empty(n_layers, seq_len, d_model, device=device, dtype=dtype)
    k_mem = torch.empty_like(q_mem)
    v_mem = torch.empty_like(q_mem)

    q_cached      = [False]*n_layers
    k_valid = torch.zeros(n_layers, seq_len, dtype=torch.bool, device=device)
    v_valid = torch.zeros_like(k_valid)

    # 用于检测“新解码”位置
    prev_mask = torch.ones(seq_len, dtype=torch.bool, device=device)  # True = still MASK

    # ---------------- Q-proj patch ------------------
    def patch_q(lidx, lin):
        orig = lin.forward
        def fwd(self, x, *a, **kw):
            if q_cached[lidx]:
                return q_mem[lidx:lidx+1].to(x.dtype)
            out = orig(x, *a, **kw)           # (1,L,d_q)
            q_mem[lidx] = out[0].to(dtype)
            q_cached[lidx] = True
            return out
        lin.forward = types.MethodType(fwd, lin)

    # ---------------- K/V-proj patch ----------------
    def patch_kv(lidx, lin, buf, valid):
        orig = lin.forward
        def fwd(self, x, *a, **kw):
            need = ~valid[lidx]               # 哪些行要重算
            if not need.any():
                return buf[lidx:lidx+1].to(x.dtype)

            # if lidx == 0:
            #     print("layer", lidx, "recalc rows:", need.sum().item())
            out = buf[lidx].clone()           # (L,d)
            sub = x[:, need]                  # (1,U,d_model)
            proj = F.linear(sub, self.weight, self.bias)  # (1,U,d_k)
            out[need] = proj[0].to(dtype)
            buf[lidx, need]  = out[need]
            valid[lidx, need]= True
            return out.unsqueeze(0).to(x.dtype)
        lin.forward = types.MethodType(fwd, lin)

    # ---------------- Attention patch --------------
    def patch_attn(lidx, attn_mod):
        orig_attn = blk.attention  # 把整段 attention() 备份
        # 每层独享缓存 (只支持 B=1)
        blk._k_cache = None  # shape (n_kv_h, T, hs)
        blk._v_cache = None
        blk._logits = None  # shape (n_head, L, T)
        blk._sum_exp = None  # (1, nh, L, 1)
        blk._ctx = None  # (1, nh, L, hs)

        def fwd(self, q, k, v, attention_bias=None, layer_past=None, use_cache=False):
            """
            q_raw, k_raw, v_raw 形状均为 (1, L, d_model)
            """
            if layer_past is not None or use_cache:
                raise NotImplementedError

            B, L, C = q.size()
            nh = self.config.n_heads
            nkv = self.config.effective_n_kv_heads
            hs = C // nh

            # ------------ 首步：直接调用原 attention() ------------
            if self._k_cache is None:
                ctx, _ = orig_attn(q, k, v,
                                   attention_bias, None, False)

                # ==== 构造本层缓存，用于后续增量 ====
                # 1) reshape / transpose
                q = q.view(B, L, nh, hs).transpose(1, 2)  # (1, nh, L, hs)
                k = k.view(B, L, nkv, hs).transpose(1, 2)  # (1, nkv, L, hs)
                v = v.view(B, L, nkv, hs).transpose(1, 2)  # (1, nkv, L, hs)

                # 2) Rotary（必须和首步内部一致）
                if self.config.rope:
                    q, k = self.rotary_emb(q, k)

                # 3) 缓存
                self._k_cache = k.clone()  # (1, nkv, L, hs)
                self._v_cache = v.clone()
                self._logits = torch.matmul(q, k.transpose(-1, -2))  # (1, nh, L, L)

                return ctx, None  # 输出首步 ctx

            # ----------- 后续步：只增量更新新列 -------------
            else:
                # reshape / transpose
                q = q.view(B, L, nh, hs).transpose(1, 2)  # (1, nh, L, hs)
                k = k.view(B, L, nkv, hs).transpose(1, 2)  # (1, nkv, L, hs)
                v = v.view(B, L, nkv, hs).transpose(1, 2)  # (1, nkv, L, hs)
                if self.config.rope:
                    q, k = self.rotary_emb(q, k)
                scale = 1.0 / math.sqrt(hs)

                # 初始化全局 sum_exp & ctx
                if self._sum_exp is None:
                    logits_full = self._logits * scale
                    if attention_bias is not None:
                        logits_full = logits_full + attention_bias
                    exp_full = torch.exp(logits_full)  # (1,nh,L,L)
                    self._sum_exp = exp_full.sum(-1, keepdim=True)  # (1,nh,L,1)
                    self._ctx = torch.matmul(  # (1,nh,L,hs)
                        exp_full / self._sum_exp, self._v_cache)

                # 选出“本步刚解码”的列
                new_mask = (~k_valid[lidx]).squeeze(0)  # (L,)
                if new_mask.any():
                    k_new = k[:, :, new_mask]  # (1, nkv, U, hs)
                    v_new = v[:, :, new_mask]
                    self._k_cache[:, :, new_mask] = k_new
                    self._v_cache[:, :, new_mask] = v_new

                    logits_new = torch.matmul(q, k_new.transpose(-1, -2)) * scale
                    if attention_bias is not None:
                        logits_new += attention_bias[..., new_mask]
                    exp_new = torch.exp(logits_new)  # (1, nh, L, U)

                    # ---------- ④ 增量归一化 + ctx 累积 ----------
                    self._sum_exp = self._sum_exp + exp_new.sum(-1, keepdim=True)
                    prob_new = exp_new / self._sum_exp  # (1,nh,L,U)
                    self._ctx = self._ctx + torch.matmul(prob_new, v_new)

                    self._logits[..., new_mask] = logits_new.squeeze(0)  # 更新列

                # 合并头并做输出投影
                ctx_heads = self._ctx  # (1,nh,L,hs)
                ctx = ctx_heads.transpose(1, 2).contiguous().view(1, L, C)
                ctx = self.attn_out(ctx)
                return ctx, None

        blk.attention = types.MethodType(fwd, blk)

    # ---------------- 注册到模型 -------------------
    for lidx, blk in enumerate(blocks):
        # Q / K / V
        patch_q (lidx, blk.q_proj)
        patch_kv(lidx, blk.k_proj, k_mem, k_valid)
        patch_kv(lidx, blk.v_proj, v_mem, v_valid)
        # Attention matmul
        patch_attn(lidx, blk.attention)

    # ---------------- step_reset -------------------
    def step_reset(mask_tensor):
        """
        mask_tensor shape : (1, L)  True = still MASK
        将“刚解码”的行标记为 invalid，让下一步重算 K/V
        """
        nonlocal prev_mask, k_valid, v_valid  # prev_mask: bool (L,)
        still_mask = mask_tensor[0].bool()  # True = 依旧 [MASK]
        decoded_now = prev_mask & (~still_mask)  # 现在刚刚解码
        if decoded_now.any():
            k_valid[:, decoded_now] = False
            v_valid[:, decoded_now] = False
        prev_mask = still_mask  # 更新「上一步」遮罩
    return step_reset


# ─────────────────────────────────── benchmark helper

def benchmark(prompt, tokenizer, *, steps, gen_len, block_len, use_qcache):
    tag = "Q‑cache" if use_qcache else "Vanilla"
    print(f"\nLoading model for {tag} …")
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True, torch_dtype=DTYPE).to(DEVICE).eval()

    # warm‑up
    with torch.inference_mode():
        _ = model(prompt[:, :1]); torch.cuda.synchronize()

    seq_len = prompt.shape[1] + gen_len

    if use_qcache:
        step_reset = attach_qkv_full_cache(model, seq_len)
    # attach_qcache_monkey(model, prompt.shape[1] + gen_len) if use_qcache else None
    with cuda_timer(f"{tag}") as get_elapsed:
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            if use_qcache:
                out = generate_with_callback(model, prompt, steps=steps, gen_length=gen_len,
                           block_length=block_len, temperature=0., cfg_scale=0.,
                           remasking='low_confidence', step_callback=lambda step, mask: step_reset(mask))
            else:
                out = generate(model, prompt, steps=steps, gen_length=gen_len,
                               block_length=block_len, temperature=0., cfg_scale=0.,
                               remasking='low_confidence')
    # decode and show (outside timing)
    answer = tokenizer.batch_decode(out[:, prompt.shape[1]:], skip_special_tokens=True)[0]
    print(f"{tag} output → {answer}\n")
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))
    # print(prof.key_averages().table(row_limit=20))

    # free memory
    del model; torch.cuda.empty_cache()

# ─────────────────────────────────────── main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", default="Explain diffusion models briefly.")
    # ap.add_argument("--steps", type=int, default=128)
    # ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--steps", type=int, default=512)
    ap.add_argument("--gen", type=int, default=512)
    ap.add_argument("--block", type=int, default=32)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    prompt_txt = tokenizer.apply_chat_template([{"role": "user", "content": args.question}], add_generation_prompt=True, tokenize=False)
    prompt = torch.tensor(tokenizer(prompt_txt)["input_ids"], device=DEVICE).unsqueeze(0)

    benchmark(prompt, tokenizer, steps=args.steps, gen_len=args.gen, block_len=args.block, use_qcache=False)
    benchmark(prompt, tokenizer, steps=args.steps, gen_len=args.gen, block_len=args.block, use_qcache=True)

if __name__ == "__main__":
    main()
