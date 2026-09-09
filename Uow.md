senninLLM 本格版（5ファイル構成）

senninLLMの本格学習・ファインチューニング基盤となる5ファイルのコード構成です。Google ColabなどのPyTorch環境で動作します。

---

ディレクトリ構成

senninLLM/
├── config.py
├── tokenizer.py
├── model.py
├── trainer.py
├── train.py
├── data/
│   ├── pretrain.txt
│   └── sft.jsonl
├── checkpoints/
└── outputs/

---

1. config.py

from dataclasses import dataclass
from pathlib import Path
import torch

@dataclass
class Config:
    project_name: str = "senninLLM"
    vocab_size: int = 32000
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_kv_head: int = 4
    n_embd: int = 768
    intermediate_size: int = 2048
    dropout: float = 0.0
    rope_theta: float = 10000.0

    batch_size: int = 2
    grad_accum_steps: int = 8

    pretrain_steps: int = 10000
    sft_steps: int = 5000

    eval_interval: int = 250
    eval_batches: int = 30
    save_interval: int = 1000

    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    weight_decay: float = 0.1
    warmup_steps: int = 500
    grad_clip: float = 1.0

    seed: int = 42

    data_dir: str = "/content/senninLLM/data"
    checkpoint_dir: str = "/content/senninLLM/checkpoints"
    output_dir: str = "/content/senninLLM/outputs"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def amp_dtype(self):
        if self.device == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    @property
    def use_amp(self):
        return self.device == "cuda"

CFG = Config()

Path(CFG.data_dir).mkdir(parents=True, exist_ok=True)
Path(CFG.checkpoint_dir).mkdir(parents=True, exist_ok=True)
Path(CFG.output_dir).mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("senninLLM")
print("=" * 60)
print(f"device: {CFG.device}")
print(f"vocab_size: {CFG.vocab_size}")
print(f"block_size: {CFG.block_size}")
print(f"layers: {CFG.n_layer}")
print(f"embedding: {CFG.n_embd}")
print(f"heads: {CFG.n_head}")
print(f"kv_heads: {CFG.n_kv_head}")
print("=" * 60)

---

2. tokenizer.py

import os
import json
import sentencepiece as spm
from config import CFG

class SenninTokenizer:
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.sp = None

        if model_path is not None and os.path.exists(model_path):
            self.sp = spm.SentencePieceProcessor(model_file=model_path)

    @property
    def vocab_size(self):
        if self.sp is None:
            return CFG.vocab_size
        return self.sp.vocab_size()

    @property
    def pad_id(self):
        if self.sp is None:
            return 0
        return self.sp.pad_id()

    @property
    def bos_id(self):
        if self.sp is None:
            return 1
        return self.sp.bos_id()

    @property
    def eos_id(self):
        if self.sp is None:
            return 2
        return self.sp.eos_id()

    @property
    def unk_id(self):
        if self.sp is None:
            return 3
        return self.sp.unk_id()

    def train(self, input_file, model_prefix):
        spm.SentencePieceTrainer.train(
            input=input_file,
            model_prefix=model_prefix,
            vocab_size=CFG.vocab_size,
            model_type="bpe",
            character_coverage=0.9995,
            normalization_rule_name="nfkc",
            pad_id=0,
            unk_id=3,
            bos_id=1,
            eos_id=2,
            user_defined_symbols=[
                "<|system|>",
                "<|user|>",
                "<|assistant|>",
                "<|end|>"
            ],
            max_sentence_length=16384,
            shuffle_input_sentence=True,
            input_sentence_size=1000000,
            train_extremely_large_corpus=False,
            hard_vocab_limit=False
        )

        self.model_path = model_prefix + ".model"
        self.sp = spm.SentencePieceProcessor(
            model_file=self.model_path
        )

    def encode(
        self,
        text,
        add_bos=False,
        add_eos=False
    ):
        if self.sp is None:
            raise RuntimeError(
                "Tokenizer model is not loaded."
            )

        ids = self.sp.encode(
            text,
            out_type=int
        )

        if add_bos:
            ids.insert(0, self.bos_id)

        if add_eos:
            ids.append(self.eos_id)

        return ids

    def decode(self, ids):
        if self.sp is None:
            raise RuntimeError(
                "Tokenizer model is not loaded."
            )

        return self.sp.decode(ids)

    def save_info(self, path):
        info = {
            "model_path": self.model_path,
            "vocab_size": self.vocab_size,
            "pad_id": self.pad_id,
            "bos_id": self.bos_id,
            "eos_id": self.eos_id,
            "unk_id": self.unk_id
        }

        with open(
            path,
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                info,
                f,
                ensure_ascii=False,
                indent=2
            )

def build_tokenizer(
    pretrain_file,
    output_dir
):
    os.makedirs(
        output_dir,
        exist_ok=True
    )

    model_prefix = os.path.join(
        output_dir,
        "sennin_tokenizer"
    )

    tokenizer = SenninTokenizer()

    tokenizer.train(
        pretrain_file,
        model_prefix
    )

    tokenizer.save_info(
        os.path.join(
            output_dir,
            "tokenizer.json"
        )
    )

    return tokenizer

---

3. model.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CFG

class RMSNorm(nn.Module):
    def __init__(
        self,
        dim,
        eps=1e-6
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.ones(dim)
        )

        self.eps = eps

    def forward(self, x):
        variance = x.float().pow(2).mean(
            dim=-1,
            keepdim=True
        )

        x = x * torch.rsqrt(
            variance + self.eps
        )

        return self.weight * x.type_as(
            self.weight
        )

def rotate_half(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]

    return torch.cat(
        (-x2, x1),
        dim=-1
    )

def apply_rope(
    q,
    k,
    cos,
    sin
):
    q = (q * cos) + (
        rotate_half(q) * sin
    )

    k = (k * cos) + (
        rotate_half(k) * sin
    )

    return q, k

class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        max_position,
        theta
    ):
        super().__init__()

        inv_freq = 1.0 / (
            theta ** (
                torch.arange(
                    0,
                    dim,
                    2,
                    dtype=torch.float32
                ) / dim
            )
        )

        self.register_buffer(
            "inv_freq",
            inv_freq,
            persistent=False
        )

        self.max_position = max_position

        self.register_buffer(
            "cos_cached",
            None,
            persistent=False
        )

        self.register_buffer(
            "sin_cached",
            None,
            persistent=False
        )

    def forward(
        self,
        seq_len,
        device,
        dtype
    ):
        if (
            self.cos_cached is None
            or self.cos_cached.shape[0] < seq_len
            or self.cos_cached.device != device
        ):
            positions = torch.arange(
                seq_len,
                device=device,
                dtype=self.inv_freq.dtype
            )

            freqs = torch.outer(
                positions,
                self.inv_freq
            )

            emb = torch.cat(
                [freqs, freqs],
                dim=-1
            )

            self.cos_cached = emb.cos()
            self.sin_cached = emb.sin()

        cos = self.cos_cached[
            :seq_len
        ].to(dtype)

        sin = self.sin_cached[
            :seq_len
        ].to(dtype)

        return (
            cos.unsqueeze(0).unsqueeze(0),
            sin.unsqueeze(0).unsqueeze(0)
        )

class GQAAttention(nn.Module):
    def __init__(
        self,
        dim,
        n_head,
        n_kv_head,
        dropout
    ):
        super().__init__()

        if n_head % n_kv_head != 0:
            raise ValueError(
                "n_head must be divisible by n_kv_head"
            )

        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = dim // n_head

        self.q_proj = nn.Linear(
            dim,
            n_head * self.head_dim,
            bias=False
        )

        self.k_proj = nn.Linear(
            dim,
            n_kv_head * self.head_dim,
            bias=False
        )

        self.v_proj = nn.Linear(
            dim,
            n_kv_head * self.head_dim,
            bias=False
        )

        self.o_proj = nn.Linear(
            dim,
            dim,
            bias=False
        )

        self.dropout = dropout

        self.rope = RotaryEmbedding(
            self.head_dim,
            CFG.block_size,
            CFG.rope_theta
        )

    def forward(
        self,
        x
    ):
        B, T, C = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(
            B,
            T,
            self.n_head,
            self.head_dim
        ).transpose(1, 2)

        k = k.view(
            B,
            T,
            self.n_kv_head,
            self.head_dim
        ).transpose(1, 2)

        v = v.view(
            B,
            T,
            self.n_kv_head,
            self.head_dim
        ).transpose(1, 2)

        cos, sin = self.rope(
            T,
            x.device,
            x.dtype
        )

        q, k = apply_rope(
            q,
            k,
            cos,
            sin
        )

        repeat_factor = (
            self.n_head //
            self.n_kv_head
        )

        if repeat_factor > 1:
            k = k.repeat_interleave(
                repeat_factor,
                dim=1
            )

            v = v.repeat_interleave(
                repeat_factor,
                dim=1
            )

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=(
                self.dropout
                if self.training
                else 0.0
            ),
            is_causal=True
        )

        y = y.transpose(
            1,
            2
        ).contiguous()

        y = y.view(
            B,
            T,
            C
        )

        return self.o_proj(y)

class SwiGLU(nn.Module):
    def __init__(
        self,
        dim,
        hidden_dim
    ):
        super().__init__()

        self.gate = nn.Linear(
            dim,
            hidden_dim,
            bias=False
        )

        self.up = nn.Linear(
            dim,
            hidden_dim,
            bias=False
        )

        self.down = nn.Linear(
            hidden_dim,
            dim,
            bias=False
        )

    def forward(self, x):
        return self.down(
            F.silu(
                self.gate(x)
            ) * self.up(x)
        )

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.norm1 = RMSNorm(
            CFG.n_embd
        )

        self.attn = GQAAttention(
            CFG.n_embd,
            CFG.n_head,
            CFG.n_kv_head,
            CFG.dropout
        )

        self.norm2 = RMSNorm(
            CFG.n_embd
        )

        self.mlp = SwiGLU(
            CFG.n_embd,
            CFG.intermediate_size
        )

    def forward(self, x):
        x = x + self.attn(
            self.norm1(x)
        )

        x = x + self.mlp(
            self.norm2(x)
        )

        return x

class SenninLLM(nn.Module):
    def __init__(
        self,
        vocab_size=None
    ):
        super().__init__()

        if vocab_size is None:
            vocab_size = CFG.vocab_size

        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(
            vocab_size,
            CFG.n_embd
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock()
                for _ in range(CFG.n_layer)
            ]
        )

        self.norm = RMSNorm(
            CFG.n_embd
        )

        self.lm_head = nn.Linear(
            CFG.n_embd,
            vocab_size,
            bias=False
        )

        self.lm_head.weight = (
            self.embedding.weight
        )

        self.apply(
            self._init_weights
        )

        for name, param in self.named_parameters():
            if name.endswith(
                "o_proj.weight"
            ) or name.endswith(
                "down.weight"
            ):
                nn.init.normal_(
                    param,
                    mean=0.0,
                    std=0.02 / math.sqrt(
                        2 * CFG.n_layer
                    )
                )

    def _init_weights(
        self,
        module
    ):
        if isinstance(
            module,
            nn.Linear
        ):
            if module.weight is not self.embedding.weight:
                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=0.02
                )

            if module.bias is not None:
                nn.init.zeros_(
                    module.bias
                )

        elif isinstance(
            module,
            nn.Embedding
        ):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

    def forward(
        self,
        input_ids,
        targets=None
    ):
        x = self.embedding(
            input_ids
        )

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(
                    -1,
                    logits.size(-1)
                ),
                targets.reshape(-1),
                ignore_index=-100
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        max_new_tokens=256,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        eos_id=None
    ):
        self.eval()

        for _ in range(max_new_tokens):
            idx_cond = input_ids[
                :, -CFG.block_size:
            ]

            logits, _ = self(
                idx_cond
            )

            logits = logits[
                :, -1, :
            ]

            logits = logits / max(
                temperature,
                1e-5
            )

            if top_k is not None:
                k = min(
                    top_k,
                    logits.size(-1)
                )

                values, _ = torch.topk(
                    logits,
                    k
                )

                threshold = values[
                    :, -1
                ].unsqueeze(-1)

                logits = torch.where(
                    logits < threshold,
                    torch.full_like(
                        logits,
                        float("-inf")
                    ),
                    logits
                )

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(
                    logits,
                    descending=True
                )

                probabilities = torch.softmax(
                    sorted_logits,
                    dim=-1
                )

                cumulative = torch.cumsum(
                    probabilities,
                    dim=-1
                )

                remove = (
                    cumulative > top_p
                )

                remove[:, 1:] = remove[
                    :, :-1
                ].clone()

                remove[:, 0] = False

                sorted_logits = sorted_logits.masked_fill(
                    remove,
                    float("-inf")
                )

                logits = torch.full_like(
                    logits,
                    float("-inf")
                )

                logits.scatter_(
                    1,
                    sorted_indices,
                    sorted_logits
                )

            probabilities = torch.softmax(
                logits,
                dim=-1
            )

            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            )

            input_ids = torch.cat(
                [
                    input_ids,
                    next_token
                ],
                dim=1
            )

            if (
                eos_id is not None
                and next_token.item() == eos_id
            ):
                break

        return input_ids

def count_parameters(model):
    return sum(
        p.numel()
        for p in model.parameters()
    )

def print_model_info(model):
    total = count_parameters(
        model
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"parameters: {total:,}"
    )

    print(
        f"trainable: {trainable:,}"
    )

    print(
        f"size fp32: {total * 4 / 1024**3:.2f} GB"
    )

---

4. trainer.py

import os
import math
import time
import torch
from torch.utils.data import Dataset, DataLoader

from config import CFG

class TokenDataset(Dataset):
    def __init__(
        self,
        tokens,
        block_size
    ):
        self.tokens = torch.tensor(
            tokens,
            dtype=torch.long
        )

        self.block_size = block_size

    def __len__(self):
        return max(
            0,
            len(self.tokens)
            - self.block_size
        )

    def __getitem__(
        self,
        index
    ):
        x = self.tokens[
            index:
            index + self.block_size
        ]

        y = self.tokens[
            index + 1:
            index + 1 + self.block_size
        ]

        return x, y

class SFTDataset(Dataset):
    def __init__(
        self,
        examples,
        tokenizer,
        block_size
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.block_size = block_size

    def __len__(self):
        return len(
            self.examples
        )

    def __getitem__(
        self,
        index
    ):
        item = self.examples[index]

        prompt = (
            "<|user|>\n"
            + item["user"]
            + "\n"
            "<|assistant|>\n"
        )

        answer = item["assistant"]

        prompt_ids = self.tokenizer.encode(
            prompt
        )

        answer_ids = self.tokenizer.encode(
            answer,
            add_eos=True
        )

        ids = (
            prompt_ids
            + answer_ids
        )

        ids = ids[
            :self.block_size
        ]

        input_ids = ids[:-1]

        target_ids = ids[1:]

        prompt_len = max(
            0,
            min(
                len(prompt_ids) - 1,
                len(target_ids)
            )
        )

        target_ids = (
            [-100] * prompt_len
            + target_ids[prompt_len:]
        )

        while len(input_ids) < self.block_size - 1:
            input_ids.append(
                self.tokenizer.pad_id
            )

            target_ids.append(
                -100
            )

        return (
            torch.tensor(
                input_ids,
                dtype=torch.long
            ),
            torch.tensor(
                target_ids,
                dtype=torch.long
            )
        )

def get_lr(
    step,
    max_lr,
    min_lr,
    warmup_steps,
    total_steps
):
    if step < warmup_steps:
        return max_lr * (
            step + 1
        ) / max(
            1,
            warmup_steps
        )

    if step >= total_steps:
        return min_lr

    progress = (
        step - warmup_steps
    ) / max(
        1,
        total_steps - warmup_steps
    )

    cosine = (
        0.5
        * (
            1
            + math.cos(
                math.pi * progress
            )
        )
    )

    return (
        min_lr
        + (
            max_lr - min_lr
        ) * cosine
    )

def create_optimizer(
    model
):
    decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if (
            param.ndim >= 2
            and "norm" not in name.lower()
            and "embedding" not in name.lower()
        ):
            decay.append(param)
        else:
            no_decay.append(param)

    optimizer = torch.optim.AdamW(
        [
            {
                "params": decay,
                "weight_decay": CFG.weight_decay
            },
            {
                "params": no_decay,
                "weight_decay": 0.0
            }
        ],
        lr=CFG.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8
    )

    return optimizer

def save_checkpoint(
    model,
    optimizer,
    step,
    loss,
    path
):
    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "loss": loss,
            "config": CFG.__dict__
        },
        path
    )

def load_checkpoint(
    model,
    optimizer,
    path,
    device
):
    checkpoint = torch.load(
        path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    if optimizer is not None:
        optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

    return (
        checkpoint.get(
            "step",
            0
        ),
        checkpoint.get(
            "loss",
            None
        )
    )

@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    max_batches
):
    model.eval()

    total_loss = 0.0
    count = 0

    for i, (
        x,
        y
    ) in enumerate(loader):
        if i >= max_batches:
            break

        x = x.to(
            device,
            non_blocking=True
        )

        y = y.to(
            device,
            non_blocking=True
        )

        _, loss = model(
            x,
            y
        )

        total_loss += loss.item()
        count += 1

    model.train()

    if count == 0:
        return float("inf")

    return total_loss / count

def train(
    model,
    train_loader,
    eval_loader,
    optimizer,
    steps,
    checkpoint_prefix,
    start_step=0
):
    device = CFG.device

    model.train()

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            CFG.use_amp
            and CFG.amp_dtype
            == torch.float16
        )
    )

    data_iter = iter(
        train_loader
    )

    best_eval = float("inf")

    for step in range(
        start_step,
        steps
    ):
        optimizer.zero_grad(
            set_to_none=True
        )

        total_loss = 0.0

        start_time = time.time()

        for _ in range(
            CFG.grad_accum_steps
        ):
            try:
                x, y = next(
                    data_iter
                )
            except StopIteration:
                data_iter = iter(
                    train_loader
                )

                x, y = next(
                    data_iter
                )

            x = x.to(
                device,
                non_blocking=True
            )

            y = y.to(
                device,
                non_blocking=True
            )

            with torch.autocast(
                device_type="cuda",
                dtype=CFG.amp_dtype,
                enabled=CFG.use_amp
            ):
                _, loss = model(
                    x,
                    y
                )

                loss = (
                    loss
                    / CFG.grad_accum_steps
                )

            if scaler.is_enabled():
                scaler.scale(
                    loss
                ).backward()
            else:
                loss.backward()

            total_loss += loss.item()

        if scaler.is_enabled():
            scaler.unscale_(
                optimizer
            )

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            CFG.grad_clip
        )

        if scaler.is_enabled():
            scaler.step(
                optimizer
            )

            scaler.update()
        else:
            optimizer.step()

        lr = get_lr(
            step,
            CFG.learning_rate,
            CFG.min_learning_rate,
            CFG.warmup_steps,
            steps
        )

        for group in optimizer.param_groups:
            group["lr"] = lr

        if (
            step % 20 == 0
            or step == steps - 1
        ):
            elapsed = (
                time.time()
                - start_time
            )

            print(
                f"step={step + 1}/{steps} "
                f"loss={total_loss:.4f} "
                f"lr={lr:.7f} "
                f"time={elapsed:.2f}s"
            )

        if (
            eval_loader is not None
            and (
                step % CFG.eval_interval == 0
                or step == steps - 1
            )
        ):
            eval_loss = evaluate(
                model,
                eval_loader,
                device,
                CFG.eval_batches
            )

            print(
                f"eval_loss={eval_loss:.4f}"
            )

            if eval_loss < best_eval:
                best_eval = eval_loss

                save_checkpoint(
                    model,
                    optimizer,
                    step + 1,
                    eval_loss,
                    checkpoint_prefix
                    + "_best.pt"
                )

        if (
            step % CFG.save_interval == 0
            and step > 0
        ):
            save_checkpoint(
                model,
                optimizer,
                step + 1,
                total_loss,
                checkpoint_prefix
                + f"_{step + 1}.pt"
            )

    return model

---

5. train.py

import os
import json
import random

import torch
from torch.utils.data import DataLoader

from config import CFG
from tokenizer import build_tokenizer
from tokenizer import SenninTokenizer
from model import SenninLLM
from model import print_model_info
from trainer import TokenDataset
from trainer import SFTDataset
from trainer import create_optimizer
from trainer import train

def set_seed(seed):
    random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )

def prepare_demo_data():
    pretrain_path = os.path.join(
        CFG.data_dir,
        "pretrain.txt"
    )

    sft_path = os.path.join(
        CFG.data_dir,
        "sft.jsonl"
    )

    if not os.path.exists(
        pretrain_path
    ):
        texts = [
            "人工知能は大量のデータからパターンを学習し、文章の生成や分類などを行う技術です。",
            "機械学習では、データと目的に応じてモデルを最適化します。",
            "深層学習では、多数の層を持つニューラルネットワークを使用します。",
            "Transformerは自然言語処理で広く利用されているニューラルネットワーク構造です。",
            "Self Attentionによって文章中のトークン同士の関係を計算できます。",
            "日本語の文章を処理するには、日本語を適切に分割できるTokenizerが重要です。",
            "学習データの品質はモデルの性能に大きな影響を与えます。",
            "モデルの性能を高めるためには、大量で多様なデータが必要です。",
            "推論時には入力された文章から次に続くトークンを予測します。",
            "言語モデルは過去のトークンを条件として次のトークンの確率を計算します。"
        ]

        with open(
            pretrain_path,
            "w",
            encoding="utf-8"
        ) as f:
            for _ in range(1000):
                for text in texts:
                    f.write(
                        text
                        + "\n"
                    )

    if not os.path.exists(
        sft_path
    ):
        examples = [
            {
                "user": "こんにちは",
                "assistant": "こんにちは。今日はどうしましたか？"
            },
            {
                "user": "あなたの名前は？",
                "assistant": "私の名前はsenninLLMです。"
            },
            {
                "user": "AIとは何ですか？",
                "assistant": "AIはデータからパターンを学習し、さまざまな処理を行う技術です。"
            },
            {
                "user": "Transformerとは何ですか？",
                "assistant": "TransformerはAttentionを利用して文章中の情報関係を処理するニューラルネットワークです。"
            },
            {
                "user": "機械学習について説明してください。",
                "assistant": "機械学習はデータから規則やパターンを学習して予測や分類を行う方法です。"
            }
        ]

        with open(
            sft_path,
            "w",
            encoding="utf-8"
        ) as f:
            for _ in range(500):
                for item in examples:
                    f.write(
                        json.dumps(
                            item,
                            ensure_ascii=False
                        )
                        + "\n"
                    )

    return (
        pretrain_path,
        sft_path
    )

def read_text(path):
    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        return f.read()

def read_jsonl(path):
    items = []

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            items.append(
                json.loads(line)
            )

    return items

def split_tokens(tokens):
    split = int(
        len(tokens) * 0.95
    )

    train_tokens = tokens[
        :split
    ]

    eval_tokens = tokens[
        split:
    ]

    return (
        train_tokens,
        eval_tokens
    )

def make_loader(
    dataset,
    shuffle=True
):
    return DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        shuffle=shuffle,
        drop_last=True,
        num_workers=2,
        pin_memory=(
            CFG.device == "cuda"
        )
    )

def train_pretrain(
    model,
    tokenizer,
    pretrain_path
):
    print(
        "\n=== PRETRAIN ==="
    )

    text = read_text(
        pretrain_path
    )

    tokens = tokenizer.encode(
        text,
        add_bos=True,
        add_eos=True
    )

    print(
        f"tokens: {len(tokens):,}"
    )

    train_tokens, eval_tokens = (
        split_tokens(tokens)
    )

    train_dataset = TokenDataset(
        train_tokens,
        CFG.block_size
    )

    eval_dataset = TokenDataset(
        eval_tokens,
        CFG.block_size
    )

    train_loader = make_loader(
        train_dataset,
        shuffle=True
    )

    eval_loader = make_loader(
        eval_dataset,
        shuffle=False
    )

    optimizer = create_optimizer(
        model
    )

    train(
        model,
        train_loader,
        eval_loader,
        optimizer,
        CFG.pretrain_steps,
        os.path.join(
            CFG.checkpoint_dir,
            "pretrain"
        )
    )

def train_sft(
    model,
    tokenizer,
    sft_path
):
    print(
        "\n=== SFT ==="
    )

    examples = read_jsonl(
        sft_path
    )

    random.shuffle(
        examples
    )

    split = int(
        len(examples) * 0.9
    )

    train_examples = (
        examples[:split]
    )

    eval_examples = (
        examples[split:]
    )

    train_dataset = SFTDataset(
        train_examples,
        tokenizer,
        CFG.block_size
    )

    eval_dataset = SFTDataset(
        eval_examples,
        tokenizer,
        CFG.block_size
    )

    train_loader = make_loader(
        train_dataset,
        shuffle=True
    )

    eval_loader = make_loader(
        eval_dataset,
        shuffle=False
    )

    optimizer = create_optimizer(
        model
    )

    train(
        model,
        train_loader,
        eval_loader,
        optimizer,
        CFG.sft_steps,
        os.path.join(
            CFG.checkpoint_dir,
            "sft"
        )
    )

def chat(
    model,
    tokenizer
):
    print(
        "\n=== CHAT ==="
    )

    model.eval()

    while True:
        try:
            user = input(
                "\nYou: "
            )
        except EOFError:
            break

        if user.strip().lower() in {
            "exit",
            "quit"
        }:
            break

        prompt = (
            "<|user|>\n"
            + user
            + "\n"
            "<|assistant|>\n"
        )

        ids = tokenizer.encode(
            prompt
        )

        input_ids = torch.tensor(
            [ids],
            dtype=torch.long,
            device=CFG.device
        )

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=256,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                eos_id=tokenizer.eos_id
            )

        generated = output[
            0
        ].tolist()

        text = tokenizer.decode(
            generated
        )

        if "<|assistant|>" in text:
            text = text.split(
                "<|assistant|>",
                1
            )[1]

        if "<|end|>" in text:
            text = text.split(
                "<|end|>",
                1
            )[0]

        if "<|user|>" in text:
            text = text.split(
                "<|user|>",
                1
            )[0]

        print(
            "senninLLM:",
            text.strip()
        )

def main():
    set_seed(
        CFG.seed
    )

    print(
        "PyTorch:",
        torch.__version__
    )

    print(
        "CUDA:",
        torch.cuda.is_available()
    )

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    pretrain_path, sft_path = (
        prepare_demo_data()
    )

    tokenizer_dir = os.path.join(
        CFG.output_dir,
        "tokenizer"
    )

    tokenizer_model = os.path.join(
        tokenizer_dir,
        "sennin_tokenizer.model"
    )

    if not os.path.exists(
        tokenizer_model
    ):
        print(
            "\n=== TRAIN TOKENIZER ==="
        )

        tokenizer = build_tokenizer(
            pretrain_path,
            tokenizer_dir
        )
    else:
        tokenizer = (
            SenninTokenizer(
                tokenizer_model
            )
        )

    print(
        "Tokenizer vocab:",
        tokenizer.vocab_size
    )

    model = SenninLLM(
        tokenizer.vocab_size
    )

    model = model.to(
        CFG.device
    )

    if CFG.device == "cuda":
        model = model.to(
            memory_format=torch.contiguous_format
        )

    print_model_info(
        model
    )

    train_pretrain(
        model,
        tokenizer,
        pretrain_path
    )

    train_sft(
        model,
        tokenizer,
        sft_path
    )

    final_path = os.path.join(
        CFG.checkpoint_dir,
        "senninLLM_final.pt"
    )

    torch.save(
        {
            "model": model.state_dict(),
            "vocab_size": tokenizer.vocab_size
        },
        final_path
    )

    print(
        "\nSaved:",
        final_path
    )

    chat(
        model,
        tokenizer
    )

if __name__ == "__main__":
    main()

---

Google Colabでの実行方法

1. 必要なライブラリをインストール

Google Colabで新しいセルを作り、以下を実行します。

!pip install -q torch sentencepiece

---

2. プロジェクトディレクトリを作成

!mkdir -p /content/senninLLM/data
!mkdir -p /content/senninLLM/checkpoints
!mkdir -p /content/senninLLM/outputs

---

3. 5ファイルを配置

以下の構成になるように配置します。

/content/senninLLM/
├── config.py
├── tokenizer.py
├── model.py
├── trainer.py
├── train.py
├── data/
├── checkpoints/
└── outputs/

"data/pretrain.txt" と "data/sft.jsonl" が存在しない場合は、"train.py" がデモ用データを自動生成します。

---

4. 学習を開始

!cd /content/senninLLM && python train.py

または、

%cd /content/senninLLM
!python train.py

---

5. 学習の流れ

実行すると基本的に以下の順番で処理されます。

senninLLM起動
    ↓
GPU確認
    ↓
データ確認
    ↓
Tokenizer学習
    ↓
PRETRAIN
    ↓
SFT
    ↓
senninLLM_final.pt保存
    ↓
CHAT

---

6. 学習後のチャット

学習が完了すると、

=== CHAT ===

You:

と表示されます。

例えば、

You: こんにちは

と入力すると、

senninLLM: こんにちは。今日はどうしましたか？

のように応答します。

終了する場合は、

exit

または、

quit

と入力します。

---

7. 生成されるファイル

学習後は主に以下が生成されます。

senninLLM/
├── config.py
├── tokenizer.py
├── model.py
├── trainer.py
├── train.py
├── data/
│   ├── pretrain.txt
│   └── sft.jsonl
├── checkpoints/
│   ├── pretrain_best.pt
│   ├── pretrain_1000.pt
│   ├── pretrain_2000.pt
│   ├── ...
│   ├── sft_best.pt
│   ├── sft_1000.pt
│   ├── sft_2000.pt
│   └── ...
└── outputs/
    └── tokenizer/
        ├── sennin_tokenizer.model
        ├── sennin_tokenizer.vocab
        └── tokenizer.json

最終モデルは、

/content/senninLLM/checkpoints/senninLLM_final.pt

に保存されます。

---

注意

現在の "prepare_demo_data()" が作るデータは動作確認用の小規模データです。

本格的にsenninLLMを学習させる場合は、

data/pretrain.txt

を大量で高品質な事前学習データに、

data/sft.jsonl

を大量で高品質な質問・回答データに置き換えてください。

例えばSFTデータは、

{"user":"Pythonとは何ですか？","assistant":"Pythonは汎用的に利用できるプログラミング言語です。"}
{"user":"機械学習とは何ですか？","assistant":"機械学習はデータからパターンを学習し、予測や分類などを行う技術です。"}

のようなJSONL形式にします。

---

今回修正した箇所

SentencePiece

hard_vocab_limit=False

を追加しています。

データ量が少ない状態で、

vocab_size=32000

を指定した場合でも、利用可能な語彙数に合わせてTokenizerを作れるようにしています。

SFT

元コードでは教師ラベルのマスク位置が1トークンずれており、回答の最初のトークンまで学習対象から除外される状態でした。

そのため、

len(prompt_ids)

を、

len(prompt_ids) - 1

に修正しています。

それ以外のモデル構造、設定値、学習ステップ数、データ形式、ディレクトリ構成、チャット処理などは変更していません。
