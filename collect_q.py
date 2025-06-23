"""q_logging_test.py

Single‑turn demo script for LLaDA that:
1. Replicates the official chat sample (load model, tokenizer, call `generate`).
2. Logs **per‑layer Query (Q) tensors for every diffusion step** using forward
   hooks – no need to touch `modeling_llada.py`.

Output hierarchy (created on‑the‑fly):
    q_cache/step_{t}/layer_{l}/q.bin
where each file is a `torch.FloatTensor(B, seq_len, d_model)`.

NOTE  ▸ This script patches the model's `forward` to auto‑increment a global
step counter so hooks know which diffusion step they belong to.
      ▸ Tested with `GSAI-ML/LLaDA-8B-Instruct` but should work for other checkpoints.
"""

import os
import torch
from transformers import AutoTokenizer, AutoModel
from generate import generate  # provided by LLaDA repo

# ────────────────────────────────────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────────────────────────────────────
MODEL_NAME = "GSAI-ML/LLaDA-8B-Instruct"
OUT_ROOT = "q_cache"          # directory to store Q tensors
GEN_LENGTH = 128              # tokens to generate
STEPS = 128                   # diffusion steps
BLOCK_LENGTH = 32             # = official default
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16        # match official demo

# ────────────────────────────────────────────────────────────────────────────────
# Q‑logging utilities
# ────────────────────────────────────────────────────────────────────────────────

def _save_q(step: int, layer_idx: int, q_tensor: torch.Tensor) -> None:
    """Persist Q to disk at q_cache/step_XXXX/layer_YYY/q.bin"""
    out_dir = os.path.join(OUT_ROOT, f"step_{step:04d}", f"layer_{layer_idx:03d}")
    os.makedirs(out_dir, exist_ok=True)
    torch.save(q_tensor.cpu(), os.path.join(out_dir, "q.bin"))


def _register_q_hooks(model: torch.nn.Module, step_state: dict) -> None:
    """Attach hooks on every block's `att_proj` to capture & save Q."""

    # Try to locate transformer blocks (works for LLaDA‑8B repo structure)
    try:
        blocks = model.model.transformer.blocks
    except Exception as e:
        raise ValueError(f"Could not find transformer block list on model: {e}")

    # Register hooks per layer
    for idx, blk in enumerate(blocks):
        if hasattr(blk, "att_proj"):
            proj_layer = blk.att_proj
            fused = True
        elif hasattr(blk, "q_proj"):
            proj_layer = blk.q_proj
            fused = False
        else:
            # No recognizable projection layer – skip
            continue

        def _make_hook(layer_index: int, is_fused: bool):  # closure
            def _hook(_, __, output):
                # output shape: fused ⇒ (..., 3*d), split ⇒ (..., d)
                q = (
                    output[..., : output.size(-1) // 3] if is_fused else output
                )
                _save_q(step_state["t"], layer_index, q.detach())

            return _hook

        proj_layer.register_forward_hook(_make_hook(idx, fused))


def _patch_forward(model: torch.nn.Module, step_state: dict) -> None:
    """Monkey‑patch model.forward to increment the diffusion step counter."""
    orig_forward = model.forward

    def _forward(*args, **kwargs):
        current_step = step_state["t"]  # read before forward (hooks use it)
        out = orig_forward(*args, **kwargs)
        step_state["t"] = current_step + 1
        return out

    model.forward = _forward  # type: ignore[misc]

# ────────────────────────────────────────────────────────────────────────────────
# Main (single‑turn chat)
# ────────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # Load model & tokenizer (official demo style)
    print("Loading model …")
    model = (
        AutoModel.from_pretrained(
            MODEL_NAME, trust_remote_code=True, torch_dtype=DTYPE
        )
        .to(DEVICE)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Attach Q‑logging machinery
    step_state = {"t": 0}
    _register_q_hooks(model, step_state)
    _patch_forward(model, step_state)

    # ── Single user question ────────────────────────────────────────────────
    user_input = input("Enter your question: ")

    conv = [{"role": "user", "content": user_input}]
    templated = tokenizer.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer(templated)["input_ids"]
    prompt = torch.tensor(input_ids, device=DEVICE).unsqueeze(0)

    # ── Generate ────────────────────────────────────────────────────────────
    print("Generating … (this will log Q tensors)")
    with torch.inference_mode():
        out = generate(
            model,
            prompt,
            steps=STEPS,
            gen_length=GEN_LENGTH,
            block_length=BLOCK_LENGTH,
            temperature=0.0,
            cfg_scale=0.0,
            remasking="low_confidence",
        )

    answer = tokenizer.batch_decode(out[:, prompt.shape[1]:], skip_special_tokens=True)[0]
    print("\nBot's reply:\n", answer)
    print("\nQ tensors saved under:", OUT_ROOT)


if __name__ == "__main__":
    main()
