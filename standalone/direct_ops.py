"""PyTorch/Triton inference arithmetic directly on scalar face leaves.

Flat and face use the same reduction tree. The only kernel difference is the
address calculation; compact faces also test occupancy and rank the lit leaves.
All host transfers are bounded to the tiles containing the requested rows.
"""
import math

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda.libdevice import popc


@triton.jit
def _value(values, masks, offsets, linear, valid, layout: tl.constexpr):
    if layout == 0:  # Original tensor order
        return tl.load(values + linear, valid, 0.)
    else:
        tile = linear // 256
        # [outer,inner,row,column] -> [row,column,outer,inner]
        pos = ((linear & 15) << 4) | ((linear >> 4) & 15)
        if layout == 1:  # All 256 leaves present
            return tl.load(values + tile * 256 + pos, valid, 0.)
        else:
            word_id = pos // 32
            bit = (1 << (pos & 31)).to(tl.uint32)
            rank = tl.full(linear.shape, 0, tl.uint32)
            lit = tl.full(linear.shape, False, tl.int1)
            for group in tl.static_range(8):
                word = tl.load(masks + tile * 8 + group, valid, 0).to(tl.uint32)
                rank += tl.where(group < word_id, popc(word.to(tl.int32, bitcast=True)), 0)
                below = (word & (bit - 1)).to(tl.int32, bitcast=True)
                rank += tl.where(group == word_id, popc(below), 0)
                lit |= (group == word_id) & ((word & bit) != 0)
            start = tl.load(offsets + tile, valid, 0)
            return tl.load(values + start + rank, valid & lit, 0.)


@triton.jit
def _project_kernel(states, values, masks, offsets, output, first, width: tl.constexpr,
                    rows, cols: tl.constexpr, lanes: tl.constexpr, layout: tl.constexpr):
    r = tl.program_id(0) * lanes + tl.arange(0, lanes)
    c = tl.arange(0, cols)
    token = tl.program_id(1)
    valid = (r[:, None] < rows) & (c[None, :] < width)
    linear = first + r[:, None] * width + c[None, :]
    w = _value(values, masks, offsets, linear, valid, layout)
    x = tl.load(states + token * width + c, c < width, 0.)
    dot = tl.sum(w * x[None, :], axis=1)
    tl.store(output + token * rows + r, dot, r < rows)


@triton.jit
def _element_kernel(states, values, masks, offsets, output, first, count,
                    width: tl.constexpr, block: tl.constexpr, layout: tl.constexpr, multiply: tl.constexpr):
    i = tl.program_id(0) * block + tl.arange(0, block)
    w = _value(values, masks, offsets, first + i % width, i < count, layout)
    if multiply:
        w *= tl.load(states + i, i < count, 0.)
    tl.store(output + i, w, i < count)


class DirectOps:
    """Direct projection, embedding and RMS scale for either packed store."""

    def __init__(self, store):
        if store.device.type != "cuda":
            raise ValueError("Direct kernels require CUDA")
        self.store = store
        self.face = hasattr(store, "get_face_tiles")
        self.calls = {"flat": 0, "full_face": 0, "compact_face": 0}

    def _encoded(self, name, start, stop):
        width = math.prod(self.store.entries[name]["shape"][1:])
        first = start * width
        if self.face:
            values, masks, offsets = self.store.get_face_tiles(name, first // 256, triton.cdiv(stop * width, 256))
            layout = 1 if masks is None else 2
            first %= 256
        else:
            values = self.store.get_rows(name, start, stop)
            masks = offsets = None
            layout = first = 0
        self.calls[("flat", "full_face", "compact_face")[layout]] += 1
        masks = values if masks is None else masks.view(torch.int32)
        return values, masks, values if offsets is None else offsets, first, layout

    def project(self, states, name, start=0, stop=None):
        rows, width = self.store.entries[name]["shape"]
        stop = rows if stop is None else stop
        rows = stop - start
        if states.size(-1) != width:
            raise ValueError(f"Invalid input width for {name}")
        values, masks, offsets, first, layout = self._encoded(name, start, stop)
        states = states.contiguous()
        out = states.new_empty((*states.shape[:-1], rows))
        grid = (triton.cdiv(rows, 8), states.numel() // width)
        _project_kernel[grid](states, values, masks, offsets, out, first, width, rows,
                              triton.next_power_of_2(width), 8, layout, enable_fp_fusion=False)
        return out

    def embedding_row(self, name, token):
        width = self.store.entries[name]["shape"][1]
        values, masks, offsets, first, layout = self._encoded(name, token, token + 1)
        out = torch.empty((1, width), dtype=torch.float32, device=self.store.device)
        _element_kernel[(triton.cdiv(width, 128),)](out, values, masks, offsets, out, first,
                                                   width, width, 128, layout, False)
        return out

    def scale(self, states, name):
        width = self.store.entries[name]["shape"][0]
        values, masks, offsets, first, layout = self._encoded(name, 0, width)
        states = states.contiguous()
        out = torch.empty_like(states)
        _element_kernel[(triton.cdiv(states.numel(), 128),)](
            states, values, masks, offsets, out, first, states.numel(), width, 128, layout, True)
        return out
