# Copyright (c) 2026, NVIDIA CORPORATION. Licensed under the Apache License, Version 2.0 (the "License") you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

"""
Pure-PyTorch reimplementation of MolMIM (molmim_70m_24_3) inference.

This loads the official MolMIM .nemo checkpoint and runs encode (-> z_mean
latent), decode, and seed sampling using only PyTorch -- no NeMo / Megatron /
apex / TransformerEngine. That makes it run natively on aarch64 + Blackwell
(GB200), where the framework stack can't currently be built/run.

Architecture (from model_config.yaml + checkpoint): Megatron T5-style encoder-
decoder with a Perceiver encoder bottleneck (hidden_steps=1 -> a single 512-d
latent token) and a MIM linear head (hiddens_to_mean) producing z_mean.
"""

from __future__ import annotations

import math
import os
import re
import tarfile
import tempfile
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

H = 512
N_HEADS = 8
HEAD_DIM = H // N_HEADS  # 64
FFN = 2048
N_LAYERS = 6
VOCAB = 640
MAX_POS = 128
EPS = 1e-5


# ---------------------------------------------------------------------------
# Tokenizer (NeMo RegExTokenizer equivalent)
# ---------------------------------------------------------------------------
class RegexTokenizer:
    def __init__(self, regex: str, vocab: List[str]):
        self.regex = re.compile("(" + regex + "|.)")
        self.vocab = {tok: i for i, tok in enumerate(vocab)}
        self.decode_vocab = {i: tok for tok, i in self.vocab.items()}
        self.pad_id, self.unk_id, self.bos_id, self.eos_id = 0, 1, 2, 3
        self.bos_token, self.eos_token = "^", "&"

    def text_to_ids(self, text: str) -> List[int]:
        return [self.vocab.get(t, self.unk_id) for t in self.regex.findall(text)]

    def ids_to_text(self, ids: List[int]) -> str:
        toks = [self.decode_vocab.get(i, "") for i in ids]
        # strip leading bos, cut at eos (mirrors RegExTokenizer.tokens_to_text)
        if toks and toks[0] == self.bos_token:
            toks = toks[1:]
        if self.eos_token in toks:
            toks = toks[: toks.index(self.eos_token)]
        return "".join(toks)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(H, FFN)
        self.fc2 = nn.Linear(FFN, H)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class Attention(nn.Module):
    """Megatron-style attention with per-head interleaved QKV."""

    def __init__(self, cross: bool):
        super().__init__()
        self.cross = cross
        if cross:
            self.query = nn.Linear(H, H)
            self.key_value = nn.Linear(H, 2 * H)
        else:
            self.query_key_value = nn.Linear(H, 3 * H)
        self.dense = nn.Linear(H, H)

    def _split_heads(self, t, chunks):
        # t: [B, T, chunks*H] -> [B, T, n_heads, chunks*head_dim] -> tuple of [B,n_heads,T,head_dim]
        B, T, _ = t.shape
        t = t.view(B, T, N_HEADS, chunks * HEAD_DIM)
        parts = t.split(HEAD_DIM, dim=-1)
        return [p.transpose(1, 2) for p in parts]

    def forward(self, x, mem=None, attn_bias=None, causal=False):
        if self.cross:
            (q,) = self._split_heads(self.query(x), 1)
            k, v = self._split_heads(self.key_value(mem), 2)
        else:
            q, k, v = self._split_heads(self.query_key_value(x), 3)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(HEAD_DIM)
        if causal:
            Tq, Tk = scores.shape[-2], scores.shape[-1]
            cmask = torch.triu(torch.ones(Tq, Tk, device=scores.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(cmask, float("-inf"))
        if attn_bias is not None:
            scores = scores + attn_bias  # [B,1,1,Tk] additive
        p = torch.softmax(scores, dim=-1)
        ctx = torch.matmul(p, v)  # [B,n_heads,Tq,head_dim]
        B, _, Tq, _ = ctx.shape
        ctx = ctx.transpose(1, 2).reshape(B, Tq, H)
        return self.dense(ctx)


class EncoderLayer(nn.Module):
    """Pre-LN: self-attn + MLP."""

    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(H, eps=EPS)
        self.self_attn = Attention(cross=False)
        self.ln2 = nn.LayerNorm(H, eps=EPS)
        self.mlp = MLP()

    def forward(self, x):
        x = x + self.self_attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class DecoderLayer(nn.Module):
    """Pre-LN: causal self-attn + cross-attn + MLP."""

    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(H, eps=EPS)
        self.self_attn = Attention(cross=False)
        self.ln2 = nn.LayerNorm(H, eps=EPS)
        self.cross_attn = Attention(cross=True)
        self.ln3 = nn.LayerNorm(H, eps=EPS)
        self.mlp = MLP()

    def forward(self, x, mem, mem_bias=None, causal=True):
        x = x + self.self_attn(self.ln1(x), causal=causal)
        x = x + self.cross_attn(self.ln2(x), mem=mem, attn_bias=mem_bias)
        x = x + self.mlp(self.ln3(x))
        return x


class PerceiverEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.init_hidden = nn.Parameter(torch.zeros(1, H))
        self.cross = nn.ModuleList([DecoderLayer() for _ in range(N_LAYERS)])
        self.self_ = nn.ModuleList([EncoderLayer() for _ in range(N_LAYERS)])
        self.final_ln = nn.LayerNorm(H, eps=EPS)

    def forward(self, enc_input, mem_bias):
        B = enc_input.shape[0]
        h = self.init_hidden.unsqueeze(0).expand(B, -1, -1)  # [B,1,H]
        for i in range(N_LAYERS):
            r = h
            h = self.cross[i](h, mem=enc_input, mem_bias=mem_bias, causal=False)
            h = self.self_[i](h)
            h = h + r
        return self.final_ln(h)


# ---------------------------------------------------------------------------
# MolMIM
# ---------------------------------------------------------------------------
class MolMIM(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_word = nn.Embedding(VOCAB, H)
        self.enc_pos = nn.Embedding(MAX_POS, H)
        self.dec_word = nn.Embedding(VOCAB, H)
        self.dec_pos = nn.Embedding(MAX_POS, H)
        self.encoder = PerceiverEncoder()
        self.hiddens_to_mean = nn.Linear(H, H)
        self.decoder = nn.ModuleList([DecoderLayer() for _ in range(N_LAYERS)])
        self.dec_final_ln = nn.LayerNorm(H, eps=EPS)
        self.tokens_head = nn.Linear(H, VOCAB)

    # -- encode ----------------------------------------------------------
    def encode(self, token_ids, enc_mask) -> torch.Tensor:
        B, L = token_ids.shape
        pos = torch.arange(L, device=token_ids.device)
        emb = self.enc_word(token_ids) + self.enc_pos(pos)[None]
        # additive mask over encoder positions: [B,1,1,L]
        mem_bias = (1.0 - enc_mask.float())[:, None, None, :] * -1e9
        perc = self.encoder(emb, mem_bias)  # [B,1,H]
        return self.hiddens_to_mean(perc)  # z_mean [B,1,H]

    # -- decode ----------------------------------------------------------
    @torch.no_grad()
    def decode(self, z_mean, max_len=MAX_POS, method="greedy", temperature=1.0,
               top_k=0, top_p=0.9, bos_id=2, eos_id=3):
        B = z_mean.shape[0]
        device = z_mean.device
        mem = z_mean  # [B,1,H] memory of length 1, no mask needed
        tokens = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        for t in range(max_len - 1):
            L = tokens.shape[1]
            pos = torch.arange(L, device=device)
            x = self.dec_word(tokens) + self.dec_pos(pos)[None]
            for layer in self.decoder:
                x = layer(x, mem=mem, mem_bias=None, causal=True)
            x = self.dec_final_ln(x)
            logits = self.tokens_head(x[:, -1, :])  # [B,vocab]
            if method == "greedy":
                nxt = logits.argmax(dim=-1)
            else:  # topkp sampling
                nxt = _sample_topkp(logits, temperature, top_k, top_p)
            nxt = torch.where(finished, torch.full_like(nxt, eos_id), nxt)
            tokens = torch.cat([tokens, nxt[:, None]], dim=1)
            finished = finished | (nxt == eos_id)
            if bool(finished.all()):
                break
        return tokens


def _sample_topkp(logits, temperature, top_k, top_p):
    logits = logits / max(temperature, 1e-6)
    if top_k and top_k > 0:
        kth = torch.topk(logits, top_k, dim=-1).values[:, -1, None]
        logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)
    if top_p and 0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = probs > top_p
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1)[:, 0]


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------
def _ld(model: MolMIM, sd: dict):
    P = "enc_dec_model."
    E = P + "enc_dec_model."

    def cp(dst: torch.Tensor, key: str):
        src = sd[key]
        assert tuple(dst.shape) == tuple(src.shape), f"{key}: {tuple(src.shape)} vs {tuple(dst.shape)}"
        dst.data.copy_(src.float())

    cp(model.enc_word.weight, P + "encoder_embedding.word_embeddings.weight")
    cp(model.enc_pos.weight, P + "encoder_embedding.position_embeddings.weight")
    cp(model.dec_word.weight, P + "decoder_embedding.word_embeddings.weight")
    cp(model.dec_pos.weight, P + "decoder_embedding.position_embeddings.weight")
    cp(model.encoder.init_hidden, E + "encoder.init_hidden")
    cp(model.tokens_head.weight, P + "tokens_head.weight")
    cp(model.tokens_head.bias, P + "tokens_head.bias")
    cp(model.hiddens_to_mean.weight, E + "hiddens_module.hidden_transforms.0.hiddens_to_mean.weight")
    cp(model.hiddens_to_mean.bias, E + "hiddens_module.hidden_transforms.0.hiddens_to_mean.bias")
    cp(model.encoder.final_ln.weight, E + "encoder.final_layernorm.weight")
    cp(model.encoder.final_ln.bias, E + "encoder.final_layernorm.bias")
    cp(model.dec_final_ln.weight, E + "decoder.model.final_layernorm.weight")
    cp(model.dec_final_ln.bias, E + "decoder.model.final_layernorm.bias")

    def load_self(layer: EncoderLayer, base: str):
        cp(layer.ln1.weight, base + "input_layernorm.weight"); cp(layer.ln1.bias, base + "input_layernorm.bias")
        cp(layer.self_attn.query_key_value.weight, base + "self_attention.query_key_value.weight")
        cp(layer.self_attn.query_key_value.bias, base + "self_attention.query_key_value.bias")
        cp(layer.self_attn.dense.weight, base + "self_attention.dense.weight")
        cp(layer.self_attn.dense.bias, base + "self_attention.dense.bias")
        cp(layer.ln2.weight, base + "post_attention_layernorm.weight"); cp(layer.ln2.bias, base + "post_attention_layernorm.bias")
        cp(layer.mlp.fc1.weight, base + "mlp.dense_h_to_4h.weight"); cp(layer.mlp.fc1.bias, base + "mlp.dense_h_to_4h.bias")
        cp(layer.mlp.fc2.weight, base + "mlp.dense_4h_to_h.weight"); cp(layer.mlp.fc2.bias, base + "mlp.dense_4h_to_h.bias")

    def load_dec(layer: DecoderLayer, base: str):
        cp(layer.ln1.weight, base + "input_layernorm.weight"); cp(layer.ln1.bias, base + "input_layernorm.bias")
        cp(layer.self_attn.query_key_value.weight, base + "self_attention.query_key_value.weight")
        cp(layer.self_attn.query_key_value.bias, base + "self_attention.query_key_value.bias")
        cp(layer.self_attn.dense.weight, base + "self_attention.dense.weight")
        cp(layer.self_attn.dense.bias, base + "self_attention.dense.bias")
        cp(layer.ln2.weight, base + "post_attention_layernorm.weight"); cp(layer.ln2.bias, base + "post_attention_layernorm.bias")
        cp(layer.cross_attn.query.weight, base + "inter_attention.query.weight")
        cp(layer.cross_attn.query.bias, base + "inter_attention.query.bias")
        cp(layer.cross_attn.key_value.weight, base + "inter_attention.key_value.weight")
        cp(layer.cross_attn.key_value.bias, base + "inter_attention.key_value.bias")
        cp(layer.cross_attn.dense.weight, base + "inter_attention.dense.weight")
        cp(layer.cross_attn.dense.bias, base + "inter_attention.dense.bias")
        cp(layer.ln3.weight, base + "post_inter_attention_layernorm.weight"); cp(layer.ln3.bias, base + "post_inter_attention_layernorm.bias")
        cp(layer.mlp.fc1.weight, base + "mlp.dense_h_to_4h.weight"); cp(layer.mlp.fc1.bias, base + "mlp.dense_h_to_4h.bias")
        cp(layer.mlp.fc2.weight, base + "mlp.dense_4h_to_h.weight"); cp(layer.mlp.fc2.bias, base + "mlp.dense_4h_to_h.bias")

    for i in range(N_LAYERS):
        load_dec(model.encoder.cross[i], E + f"encoder.cross_attn_layers.{i}.layers.0.")
        load_self(model.encoder.self_[i], E + f"encoder.self_attn_layers.{i}.layers.0.")
        load_dec(model.decoder[i], E + f"decoder.model.layers.{i}.")


class MolMIMRunner:
    """High-level API mirroring MolMIMInference.seq_to_hiddens/hiddens_to_seq/sample."""

    def __init__(self, nemo_path: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        regex, vocab, sd = self._extract(nemo_path)
        self.tok = RegexTokenizer(regex, vocab)
        self.model = MolMIM().to(self.device).eval()
        _ld(self.model, sd)
        self.model.to(self.device)

    @staticmethod
    def _extract(nemo_path: str):
        with tempfile.TemporaryDirectory() as d:
            with tarfile.open(nemo_path) as tar:
                tar.extractall(d)
            files = os.listdir(d)
            vocab_f = next(f for f in files if f.endswith(".vocab"))
            model_f = next(f for f in files if f.endswith(".model"))
            ckpt_f = next(f for f in files if f.endswith(".ckpt"))
            with open(os.path.join(d, vocab_f)) as fh:
                vocab = [ln.rstrip("\n") for ln in fh]
            with open(os.path.join(d, model_f)) as fh:
                regex = fh.read().strip()
            obj = torch.load(os.path.join(d, ckpt_f), map_location="cpu", weights_only=False)
            sd = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj
        return regex, vocab, sd

    @torch.no_grad()
    def seq_to_hiddens(self, smiles: List[str]) -> torch.Tensor:
        ids = [self.tok.text_to_ids(s) for s in smiles]
        L = max(len(x) for x in ids)
        tok = torch.full((len(ids), L), self.tok.pad_id, dtype=torch.long)
        mask = torch.zeros(len(ids), L, dtype=torch.long)
        for i, x in enumerate(ids):
            tok[i, : len(x)] = torch.tensor(x)
            mask[i, : len(x)] = 1
        return self.model.encode(tok.to(self.device), mask.to(self.device))  # [B,1,H]

    @torch.no_grad()
    def hiddens_to_seq(self, hiddens: torch.Tensor, method="greedy", **kw) -> List[str]:
        toks = self.model.decode(hiddens.to(self.device), method=method, **kw)
        return [self.tok.ids_to_text(row.tolist()) for row in toks]

    @torch.no_grad()
    def sample(self, smiles: str, num_samples: int = 10, scaled_radius: float = 1.0,
               method="greedy") -> List[str]:
        z = self.seq_to_hiddens([smiles])  # [1,1,H]
        z = z.repeat_interleave(num_samples, 0)
        z = z + scaled_radius * torch.randn_like(z)
        return self.hiddens_to_seq(z, method=method)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
        "MOLMIM_CHECKPOINT", os.path.expanduser("~/molmim_dl/molmim_v1.3/molmim_70m_24_3.nemo"))
    r = MolMIMRunner(path)
    tests = [
        "c1ccccc1",
        "CCO",
        "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
        "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5",  # imatinib
    ]
    print("device:", r.device)
    z = r.seq_to_hiddens(tests)
    print("hiddens shape:", tuple(z.shape))
    recon = r.hiddens_to_seq(z, method="greedy")
    ok = 0
    for s, rc in zip(tests, recon):
        match = (s == rc)
        ok += int(match)
        print(f"[{'OK ' if match else 'DIFF'}] in : {s}\n        out: {rc}")
    print(f"round-trip exact: {ok}/{len(tests)}")
    print("--- sample (radius=1.0) around aspirin ---")
    for s in r.sample("CC(=O)Oc1ccccc1C(=O)O", num_samples=5, scaled_radius=1.0):
        print("  ", s)
