"""Inference for the released FP32 TinyMoE checkpoint using a named weight store.

The store provides get(original_tensor_name) -> float32 tensor on the target
device. This executor owns the complete inference graph, including routing.
It implements the checkpoint's existing Mixtral architecture using PyTorch.
"""
import torch
from torch.nn import functional as F


class Executor:
    def __init__(self, store, config, device="cuda", row_window=0, direct=False):
        if config["model_type"] != "mixtral" or config["hidden_act"] != "silu":
            raise ValueError("Expected a Mixtral checkpoint with SiLU experts")
        if row_window < 0:
            raise ValueError("row_window must be nonnegative")
        self.direct = None
        if direct:
            if __package__:
                from .direct_ops import DirectOps
            else:
                from direct_ops import DirectOps
            self.direct = DirectOps(store)
        self.row_window = row_window
        self.vocab_size = config["vocab_size"]
        self.store = store
        self.last_routes = []
        self.width = config["hidden_size"]
        self.heads = config["num_attention_heads"]
        self.head_dim = config.get("head_dim") or self.width // self.heads
        self.repeats = self.heads // config["num_key_value_heads"]
        self.layers = config["num_hidden_layers"]
        self.experts = config["num_local_experts"]
        self.top_k = config["num_experts_per_tok"]
        self.eps = config["rms_norm_eps"]
        self.window = config["sliding_window"]
        # The reference checkpoint initializes inverse frequencies in FP32 on CPU.
        exponent = torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim
        self.inv_frequency = (1.0 / (config["rope_theta"] ** exponent)).to(device)

    def _norm(self, hidden, name):
        scale = torch.rsqrt(hidden.square().mean(-1, keepdim=True) + self.eps)
        normalized = hidden * scale
        if self.direct is not None:
            return self.direct.scale(normalized, name)
        return self.store.get(name) * normalized

    def _project(self, hidden, name):
        if self.direct is not None:
            return self.direct.project(hidden, name + ".weight")
        return F.linear(hidden, self.store.get(name + ".weight"))

    @staticmethod
    def _rotate(tensor, cos, sin):
        first, second = tensor.chunk(2, dim=-1)
        return tensor * cos + torch.cat((-second, first), dim=-1) * sin

    def _attention(self, hidden, prefix, cos, sin, mask, previous):
        batch, tokens, _ = hidden.shape
        shape = (batch, tokens, -1, self.head_dim)
        query = self._project(hidden, prefix + "q_proj").view(shape).transpose(1, 2)
        key = self._project(hidden, prefix + "k_proj").view(shape).transpose(1, 2)
        value = self._project(hidden, prefix + "v_proj").view(shape).transpose(1, 2)
        query = self._rotate(query, cos, sin)
        key = self._rotate(key, cos, sin)

        if previous is None:
            key = key.contiguous()
            value = value.contiguous()
        else:
            key = torch.cat((previous[0], key), dim=-2)
            value = torch.cat((previous[1], value), dim=-2)
        cache_start = max(0, key.size(-2) - self.window + 1)
        layer_cache = (key[:, :, cache_start:, :], value[:, :, cache_start:, :])
        key_heads = key.repeat_interleave(self.repeats, dim=1)
        value_heads = value.repeat_interleave(self.repeats, dim=1)
        scores = (query @ key_heads.transpose(-1, -2)) * self.head_dim**-0.5
        probs = (scores + mask).softmax(dim=-1)
        attention = (probs @ value_heads).transpose(1, 2).contiguous().view(batch, tokens, -1)
        return self._project(attention, prefix + "o_proj"), layer_cache

    def _mixture(self, hidden, prefix):
        states = hidden.reshape(-1, self.width)
        probabilities = self._project(states, prefix + "gate").softmax(dim=-1)
        weights, indices = probabilities.topk(self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        assigned = F.one_hot(indices, self.experts).permute(2, 1, 0)
        active = assigned.sum(dim=(1, 2)).nonzero().flatten().tolist()
        self.last_routes.append(active)
        combined = torch.zeros_like(states)
        for expert in active:
            slot, row = torch.where(assigned[expert])
            name = f"{prefix}experts.{expert}."
            # The eager reference evaluates gate and up in one fused projection.
            if self.direct is None:
                gate_up = torch.cat((self.store.get(name + "w1.weight"), self.store.get(name + "w3.weight")))
                gate, up = F.linear(states[row], gate_up).chunk(2, dim=-1)
                del gate_up
            else:
                active_states = states[row]
                gate = self._project(active_states, name + "w1")
                up = self._project(active_states, name + "w3")
            contribution = self._project(F.silu(gate) * up, name + "w2")
            combined.index_add_(0, row, contribution * weights[row, slot, None])
        return combined.reshape_as(hidden)

    def _embed(self, input_ids):
        name = "model.embed_tokens.weight"
        if not self.row_window and self.direct is None:
            return F.embedding(input_ids, self.store.get(name))
        ids, inverse = input_ids.unique(sorted=True, return_inverse=True)
        if self.direct is not None:
            rows = [self.direct.embedding_row(name, token) for token in ids.tolist()]
        else:
            rows = [self.store.get_rows(name, token, token + 1) for token in ids.tolist()]
        return F.embedding(inverse, torch.cat(rows, dim=0))

    def _logits(self, hidden):
        if not self.row_window:
            return self._project(hidden, "lm_head")
        parts = []
        for start in range(0, self.vocab_size, self.row_window):
            stop = min(self.vocab_size, start + self.row_window)
            if self.direct is not None:
                parts.append(self.direct.project(hidden, "lm_head.weight", start, stop))
            else:
                rows = self.store.get_rows("lm_head.weight", start, stop)
                parts.append(F.linear(hidden, rows))
                del rows
        return torch.cat(parts, dim=-1)

    @torch.inference_mode()
    def forward(self, input_ids, cache=None):
        """Return all next-token logits and an opaque cache for the next call.

        Input is an unpadded rectangular batch [batch, tokens]. Start each new
        generation with cache=None and pass back the returned cache to continue.
        """
        seen, previous = cache if cache is not None else (0, [None] * self.layers)
        tokens = input_ids.size(1)
        positions = torch.arange(seen, seen + tokens, device=input_ids.device)
        freqs = (self.inv_frequency[None, :, None] @ positions[None, None, :].float()).transpose(1, 2)
        angles = torch.cat((freqs, freqs), dim=-1)[:, None, :, :]
        cos, sin = angles.cos(), angles.sin()
        first_key = max(0, seen - self.window + 1)
        keys = torch.arange(first_key, seen + tokens, device=input_ids.device)
        distances = positions[:, None] - keys[None, :]
        allowed = (distances >= 0) & (distances < self.window)
        mask = torch.zeros_like(allowed, dtype=torch.float32).masked_fill_(~allowed, torch.finfo(torch.float32).min)

        hidden = self._embed(input_ids)
        self.last_routes = []
        updated_cache = []
        for layer in range(self.layers):
            prefix = f"model.layers.{layer}."
            normalized = self._norm(hidden, prefix + "input_layernorm.weight")
            attended, layer_cache = self._attention(normalized, prefix + "self_attn.", cos, sin, mask, previous[layer])
            updated_cache.append(layer_cache)
            hidden = hidden + attended
            normalized = self._norm(hidden, prefix + "post_attention_layernorm.weight")
            hidden = hidden + self._mixture(normalized, prefix + "block_sparse_moe.")
        logits = self._logits(self._norm(hidden, "model.norm.weight"))
        return logits, (seen + tokens, updated_cache)
