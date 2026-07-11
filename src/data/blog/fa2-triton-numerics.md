---
title: "Building FlashAttention-2 in Triton: A Numerics Story"
pubDatetime: 2026-07-11T12:00:00Z
description: "Implementing FlashAttention-2 forward and backward in Triton from the papers — O(n) memory verified, beats PyTorch SDPA in fp32 on H100, and an honest post-mortem of a benchmarking misdiagnosis."
tags:
  - triton
  - flashattention
  - gpu-kernels
  - numerics
  - benchmarking
draft: false
---

## TL;DR

I built FlashAttention-2 — forward and backward, causal and non-causal — in Triton, from the
papers, and benchmarked it against naive attention and PyTorch's fused SDPA on an H100. The
kernel is O(n) memory (naive OOMs an 80 GB card at fp32/seq-8192; mine runs it in 36 ms),
beats SDPA by ~1.8× in fp32 at long sequence, and sits ~2.3× behind SDPA in bf16 — headroom
I can name, not mystery. Along the way I misdiagnosed my own kernel ("it's not using tensor
cores") from a bad benchmark, and the correction taught me more about performance measurement
than the kernel did. FP8 attention is designed and queued as the next step.

## 1. Why attention is a memory problem, not a FLOPs problem

Naive attention materializes the n×n score matrix. That's O(n²) memory, and — worse — O(n²)
HBM traffic, and on a modern GPU bandwidth is the scarce resource, not FLOPs.

Measured on an H100 80GB (batch=8, heads=8, causal): naive peak memory grows ~4× per 2× of
sequence length — 122 MB → 874 MB → 12.5 GB → 49.6 GB for bf16, d_head=64. At fp32/seq=8192
the naive implementation **OOMs the entire 80 GB card** (the n×n scores are ~17 GB per copy,
and forward+backward needs several), while a flash implementation runs the same shape in
16.6 ms using 2.3 GB.

FlashAttention's answer: never form S at all. Stream K/V tiles through SRAM and keep a
running softmax. The rest of this post is what "running softmax" means and what it took to
make the backward pass work without the matrix it never stored.

## 2. The one trick: online softmax

Softmax is the only thing stopping attention from being a simple tiled matmul, because its
denominator needs the whole row at once. Online softmax (Milakov & Gimelshein, 2018) turns
softmax from a whole-row operation into a streaming one.

Keep two running quantities per row: the running max `m` and the running denominator `ℓ`.
When a new tile of scores arrives with tile-max `m_new`:

```
m' = max(m, m_new)
ℓ  = ℓ · exp(m − m') + rowsum(exp(S_tile − m'))
```

The factor `exp(m − m')` is the whole trick: every previously accumulated exponential was
computed relative to the old max, so when a bigger max appears, one multiply re-references
all of it to the new max. For attention you carry one more accumulator — the running output
`O` — and it gets rescaled by the same factor before adding the new tile's contribution:

```
O = O · exp(m − m') + exp(S_tile − m') @ V_tile
```

Divide by `ℓ` once, at the very end. After the last tile the result is *exactly* the
full-row softmax — not an approximation — numerically stable (everything is max-subtracted),
and the n×n score matrix never exists. That single recurrence is FlashAttention; everything
else is engineering around it.

## 3. Building it: forward, then backward

**Forward** (FA2 Algorithm 1): one Triton program instance owns one query block for one
(batch, head) — the launch grid runs over the *sequence* dimension, which is FA2's key
parallelism change over FA1. The Q block is loaded into SRAM once and stays; K/V blocks
stream past it; the program carries (m, ℓ, O) exactly as in §2, and stores two things at the
end: the normalized output O and the logsumexp L = m + log ℓ — one scalar per query row.

**Causal masking** costs almost nothing if you do it at the block level: key blocks entirely
in the future are skipped outright (roughly halving the work), and only the diagonal block
needs an elementwise mask. One numerics detail: mask with −1e6, not −inf — in bf16, −inf can
turn into NaN through the rescale arithmetic, while −1e6 just exponentiates to a clean zero.

**Backward** (FA2 Algorithm 2) is where the "never store the matrix" bill comes due: the
gradients need the probability matrix P, and it was never written down. The recompute trick:
that stored logsumexp L is all you need — recompute S tile-by-tile and P = exp(S − L)
directly, no second normalization pass. One more identity makes it tile cleanly: the softmax
Jacobian's per-row correction Σⱼ P·dP collapses to D = rowsum(O ∘ dO), computable from
tensors you have without any n×n object. The kernel computes P twice — once for dQ, once for
dK/dV — so each thread block owns its output tile and no cross-block atomics are needed:
extra FLOPs, cheaper than synchronization.

Timeline, honestly: forward passed the reference tests on day 2 of kernel week, ahead of my
own schedule; causal masking and the full backward landed over the following weekend. All
four test variants (forward/backward × causal/non-causal) pass against a PyTorch reference.

## 4. Results (H100 80GB SXM5 · torch 2.11/cu130 · triton 3.6.0)

Config: batch=8, heads=16, head_dim=64, non-causal, median of 100 iterations after warmup.

### Forward latency @ seq=8192 (ms)

| dtype | naive | PyTorch SDPA | my fa2_triton |
|---|--:|--:|--:|
| bf16 | OOM* | ~4.5 | 10.6 |
| fp32 | OOM | 64.4 | **36.3** |

*naive OOMs at 8192; at seq=4096 it needs ~25 GB where fa2 uses 452 MB — ~55× more memory.

### fa2_triton forward + backward scaling (bf16)

| seq | fwd ms | bwd ms | peak MB |
|--:|--:|--:|--:|
| 2048 | 0.77 | 3.36 | 226 |
| 4096 | 2.76 | 12.11 | 452 |
| 8192 | 10.61 | 46.30 | 904 |

Three things these numbers say:

1. **The O(n) memory claim is real and survives the backward.** Peak memory doubles per 2×
   sequence — the recompute strategy holds end-to-end.
2. **A hand-written Triton kernel beats PyTorch's fused attention in fp32** (~1.8× at 8192).
3. **bf16 trails SDPA ~2.3×.** That gap is named headroom — tile-size tuning and non-matmul
   op reduction — not a mystery (see §5 for why I can say that with confidence).

Backward runs ≈ 3.5–4.5× the forward — the expected ratio for a recompute-based backward
(more matmuls plus re-materializing P).

## 5. The debugging story: the "missing tensor cores" that weren't

My first benchmark of the forward kernel produced a strange, clean-looking signal: bf16 and
fp32 latency were nearly identical (39.7 vs 36.3 ms at seq=8192). SDPA, on the same sweep,
was ~14× faster in bf16 than fp32. The inference wrote itself: *my matmuls aren't dispatching
to bf16 tensor cores — they're running at fp32 throughput regardless of input dtype.* I wrote
it down as a root cause and sketched the fix (force bf16 MMA with fp32 accumulation).

Then I re-benchmarked three days later, after finishing the backward pass, and the story
fell apart in the right direction: bf16 forward now ran at 10.6 ms against fp32's unchanged
36.3 — a 3.4× dtype ratio, exactly the signature of tensor cores engaging. Nothing was wrong
with the hardware path.

Here's the embarrassing part: I can't fully reconstruct what flipped it. The kernel went
through several edits that week (causal masking, the backward pass), and the first table had
been transcribed by hand from an on-screen photo of whatever build was live that morning. A
kernel fix, a stale build, a transcription of the wrong sweep — any of them fits the data,
and I didn't keep the provenance to tell them apart. The misdiagnosis cost me nothing; the
unreproducible *fix* is the part that actually stung.

Two lessons I'm keeping:

1. **A flat dtype ratio is a symptom with several causes.** "Not using tensor cores" was one
   hypothesis; a stale build, a bad transcription, a config mismatch are others. I
   architected a fix before re-measuring — backwards.
2. **Benchmark tables need provenance.** My provisional numbers were transcribed from an
   on-screen photo of an earlier build and labeled as if they were current. The fix wasn't
   just re-running — it was marking every table with exactly what produced it.

The gap that remains (2.3× vs SDPA in bf16) is now a *measured* gap on a *verified* tensor-
core path, which is what makes it actionable headroom rather than a bug hunt.

## 6. Future work: FP8 attention

The natural next step is an FP8 forward path, and it's designed: quantize Q and K to E4M3
with amax-based scaling, run QKᵀ on FP8 tensor cores accumulating in fp32, keep the softmax
in high precision, and measure the accuracy/speed trade against bf16. FlashAttention-3's
recipe adds incoherent processing — a random Hadamard rotation before quantization to spread
outliers — which is the same principle I worked with in NVFP4 quantized-training land:
disperse the outliers so a narrow format can hold the distribution. That through-line — the
same numerics idea recurring in FP4 training and FP8 attention — is why this is the piece of
the roadmap I most want to build.

## References

- Dao, Fu, Ermon, Rudra, Ré — *FlashAttention: Fast and Memory-Efficient Exact Attention
  with IO-Awareness* (2022)
- Dao — *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning* (2023)
- Shah, Bikshandi, Zhang, Thakkar, Ramani, Dao — *FlashAttention-3: Fast and Accurate
  Attention with Asynchrony and Low-precision* (2024)
- Milakov & Gimelshein — *Online normalizer calculation for softmax* (2018)
- Triton documentation & fused-attention tutorial
- My earlier post: [Notes on 4-Bit LLMs](/posts/qad-nemotron-nvfp4)
