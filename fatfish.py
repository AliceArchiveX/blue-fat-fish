"""FatFish - Spark-X2.5-4B with a persistent persona lane.

Usage:
    python fatfish.py

Requirements:
    - Spark-X2.5-4B weights (local path configured below, or set
      SPARK_MODEL_PATH env var)
    - torch, transformers, accelerate

The persona is carried entirely by the lane parameters in `lanes/` - there
are no prompts, templates or text hooks involved. Delete the `lanes/`
directory to get back the stock model behavior.
"""
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import transformers.modeling_rope_utils as _mru
from transformers import PretrainedConfig as _PC

# --- compat shim for transformers 5.x with remote-code configs -------------
_orig_validate_rope = _mru.RotaryEmbeddingConfigMixin.validate_rope


def _fixed_validate_rope(*args):
    # compatible with both validator conventions: (self) and (cls, value)
    self = args[0] if len(args) == 1 else args[1]
    rp = getattr(self, 'rope_parameters', None)
    if isinstance(rp, dict):
        for k in list(rp.keys()):
            if not isinstance(rp[k], dict):
                del rp[k]
    return self


_mru.RotaryEmbeddingConfigMixin.validate_rope = _fixed_validate_rope
_PC.__class_validators__ = tuple(
    _fixed_validate_rope if getattr(v, '__name__', '') == 'validate_rope' else v
    for v in _PC.__class_validators__)
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
MODEL_PATH = os.environ.get('SPARK_MODEL_PATH', '')
LANE_PATH = HERE / 'lanes' / 'persona_lane.pt'
LANE_LAYERS = [15, 16, 17, 18, 19]
REP_PEN = 1.2


def load_lane(mlp_map, device):
    copies = torch.load(LANE_PATH, weights_only=True)
    handles = []
    for li in LANE_LAYERS:
        c = copies[str(li)]
        up_c = c['up'].to(device).float()
        gate_c = c['gate'].to(device).float()
        down_c = c['down'].to(device).float()

        def make_lane_op(up_c, gate_c, down_c):
            def lane_forward(module, args, output):
                x = args[0].float()
                extra = F.linear(F.silu(F.linear(x, gate_c)) *
                                 F.linear(x, up_c), down_c)
                return output + extra.to(output.dtype)
            return lane_forward

        handles.append(mlp_map[li].register_forward_hook(make_lane_op(up_c, gate_c, down_c)))
    return handles


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if not MODEL_PATH or not Path(MODEL_PATH).exists():
        print(f'未找到模型目录: {MODEL_PATH}')
        print('请下载 Spark-X2.5-4B 并将路径写入 SPARK_MODEL_PATH 环境变量，')
        print('或直接修改本文件顶部的 MODEL_PATH。')
        return
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True,
                                        trust_remote_code=True)
    print('加载模型中...', flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, local_files_only=True, trust_remote_code=True,
        dtype=torch.bfloat16, device_map='auto', low_cpu_mem_usage=True).eval()
    device = model.get_input_embeddings().weight.device

    mlp_map = {}
    for name, mod in model.named_modules():
        if name.endswith('.mlp'):
            for p in name.split('.'):
                if p.isdigit():
                    mlp_map[int(p)] = mod
                    break
    load_lane(mlp_map, device)

    eos = model.generation_config.eos_token_id
    stop_ids = set(eos if isinstance(eos, list) else [eos, tok.eos_token_id])

    def build(history, user_msg):
        msgs = []
        for role, content in history[-8:]:
            msgs.append({'role': role, 'content': content})
        msgs.append({'role': 'user', 'content': user_msg})
        try:
            return tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True,
                                           enable_thinking=False)
        except Exception:
            return tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True)

    def generate(prompt_text, mx=200):
        seq = tok.encode(prompt_text, add_special_tokens=False)
        cur = torch.tensor([seq], device=device)
        cache = None
        out_ids = []
        with torch.inference_mode():
            for _ in range(mx):
                out = model(input_ids=cur, past_key_values=cache, use_cache=True)
                cache = out.past_key_values
                z = out.logits[0, -1].float()
                if out_ids:
                    for t in set(out_ids[-40:]):
                        z[t] /= REP_PEN
                nxt = int(z.argmax())
                if nxt in stop_ids:
                    break
                out_ids.append(nxt)
                cur = torch.tensor([[nxt]], device=device)
        return tok.decode(out_ids).strip()

    print('=' * 56)
    print('  大肥鱼对话框（输入 /new 重置对话，/exit 退出）')
    print('=' * 56)
    history = []
    while True:
        try:
            user = input('\n你> ').strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user:
            continue
        if user == '/exit':
            break
        if user == '/new':
            history = []
            print('（新对话已开启）')
            continue
        resp = generate(build(history, user))
        history.append(('user', user))
        history.append(('assistant', resp))
        print(f'大肥鱼> {resp}')


if __name__ == '__main__':
    main()
