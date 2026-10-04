from __future__ import annotations

import threading
from pathlib import Path


class Attention:  # replaced with nn.Module at runtime; avoids importing torch in demo mode
    pass


class KumaGPTInference:
    """Loads KumaGPT once and exposes a small, backend-neutral generate API."""

    def __init__(self, settings):
        self.settings = settings
        self._lock = threading.Lock()
        self.model = None
        self.tokenizer = None
        self.device = None

    def load(self) -> None:
        if self.settings.model_backend == "demo":
            return
        if self.settings.model_backend != "torch":
            raise RuntimeError("KUMAGPT_MODEL_BACKEND must be 'demo' or 'torch'")
        self._load_torch()

    def _load_torch(self) -> None:
        try:
            import sentencepiece as spm
            import torch
            import torch.nn as nn
            from torch.nn import functional as F
        except ImportError as exc:
            raise RuntimeError("torch と sentencepiece をインストールしてください") from exc

        model_path, tokenizer_path = Path(self.settings.model_path), Path(self.settings.tokenizer_path)
        if not model_path.is_file() or not tokenizer_path.is_file():
            raise RuntimeError(f"モデルまたはトークナイザーがありません: {model_path}, {tokenizer_path}")

        class TorchAttention(nn.Module):
            def __init__(self, d_model, d_head, h):
                super().__init__()
                if d_head % h:
                    raise ValueError("d_head must be divisible by h")
                self.Wk, self.Wq, self.Wv = nn.Linear(d_model, d_head), nn.Linear(d_model, d_head), nn.Linear(d_model, d_head)
                self.h, self.linear = h, nn.Linear(d_head, d_model)

            def forward(self, x):
                k, q, v = self.Wk(x), self.Wq(x), self.Wv(x)
                batch, length, width = k.shape
                split = lambda z: z.view(batch, length, self.h, width // self.h).transpose(1, 2)
                y = F.scaled_dot_product_attention(split(q), split(k), split(v), is_causal=True)
                return self.linear(y.transpose(1, 2).reshape(batch, length, width))

        class Block(nn.Module):
            def __init__(self, d_model, d_head, d_ff, h):
                super().__init__()
                self.ln1, self.attn, self.ln2 = nn.LayerNorm(d_model), TorchAttention(d_model, d_head, h), nn.LayerNorm(d_model)
                self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
                self.dropout = nn.Dropout(0.1)

            def forward(self, x):
                x = x + self.dropout(self.attn(self.ln1(x)))
                return x + self.dropout(self.ff(self.ln2(x)))

        class Model(nn.Module):
            def __init__(self, vocab_size, block_size, d_model, d_head, d_ff, n_layer, h):
                super().__init__()
                self.token_embedding = nn.Embedding(vocab_size, d_model)
                self.pos_embedding = nn.Embedding(block_size, d_model)
                self.blocks = nn.ModuleList([Block(d_model, d_head, d_ff, h) for _ in range(n_layer)])
                self.lnf, self.linear_output = nn.LayerNorm(d_model), nn.Linear(d_model, vocab_size)
                self.linear_output.weight = self.token_embedding.weight
                self.block_size = block_size

            def forward(self, value):
                _, length = value.shape
                x = self.token_embedding(value) + self.pos_embedding(torch.arange(length, device=value.device))
                for block in self.blocks:
                    x = block(x)
                return self.linear_output(self.lnf(x))

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        config = checkpoint["config"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # SentencePiece on Windows can reject non-ASCII file paths. Loading the
        # serialized model avoids that native path conversion entirely.
        self.tokenizer = spm.SentencePieceProcessor(model_proto=tokenizer_path.read_bytes())
        self.model = Model(**config).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def _build_prompt(self, messages: list[dict[str, str]]) -> str:
        formatted_messages = []
        for item in messages:
            if item["role"] == "assistant":
                formatted_messages.append(
                    f"<assistant>{item['content']}<|endoftext|>"
                )
            else:
                formatted_messages.append(f"<user>{item['content']}")
        return "\n".join(formatted_messages) + "\n<assistant>"

    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        top_k: int | None = None,
    ) -> str:
        if self.settings.model_backend == "demo":
            return f"（デモ応答）{messages[-1]['content']}"
        import torch
        from torch.nn import functional as F

        sampling_temperature = temperature if temperature is not None else self.settings.temperature
        sampling_top_k = top_k if top_k is not None else self.settings.top_k
        prompt = self._build_prompt(messages[-self.settings.context_message_limit :])
        token_ids = self.tokenizer.encode(prompt, out_type=int)
        token_ids = token_ids[-self.model.block_size :]
        generated: list[int] = []
        stop_ids = {self.tokenizer.piece_to_id("<user>"), self.tokenizer.piece_to_id("<|endoftext|>")}
        with self._lock, torch.inference_mode():
            current = torch.tensor([token_ids], dtype=torch.long, device=self.device)
            for _ in range(self.settings.max_new_tokens):
                logits = self.model(current[:, -self.model.block_size :])[:, -1, :] / sampling_temperature
                values, _ = torch.topk(logits, min(sampling_top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float("inf")
                next_token = torch.multinomial(F.softmax(logits, dim=-1), 1)
                token_id = next_token.item()
                if token_id in stop_ids:
                    break
                generated.append(token_id)
                current = torch.cat((current, next_token), dim=1)
        answer = self.tokenizer.decode(generated).strip()
        if not answer:
            raise RuntimeError("モデルが空の回答を生成しました")
        return answer
