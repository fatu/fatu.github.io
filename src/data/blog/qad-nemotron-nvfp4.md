---
title: "Notes on 4-Bit LLMs"
pubDatetime: 2026-06-08T12:00:00Z
description: "Notes on getting 4-bit LLMs to work — NVFP4, block scaling, and quantization-aware distillation, with a case study on Nemotron-3-Nano."
tags:
  - quantization
  - nvfp4
  - llm
  - numerics
  - blackwell
draft: true
---

## 1. TL;DR

Getting an LLM to 4 bits without breaking it is a two-axis problem: a **format** that can represent 4-bit numbers faithfully, and a **training recipe** that makes the model tolerate them.

- **The format ladder (§3).** FP8 with a per-tensor scale handles 8 bits but runs out of road at 4. MX formats share an FP8 scale across a *block* of elements; **NVFP4** — Blackwell's native E2M1 with a 16-element block scale *and* a tensor-level FP32 scale — is the rung that makes 4-bit weights work. It splits the job in two: the scale places the block on the number line, the element resolves locally.
- **The accuracy-recovery ladder (§4).** PTQ (calibrate-and-apply) is near-lossless at 8 bits and falls apart at 4. The fix is *quantization-aware training* — training through fake-quantization so the model adapts to the noise — in two sibling flavors: **QAT** (supervised by cross-entropy on hard labels) and **QAD** (supervised by distillation against a teacher's soft labels).
- **The recipe is asymmetric on purpose (§5).** On a 30B hybrid Mamba/MoE/attention model (Nemotron-3-Nano), the win comes from *spending precision where it matters*: NVFP4 weights + FP8 activations + FP8 KV cache for the bulk, with BF16/FP32 kept for the load-bearing few — embeddings, the LM head, all six attention layers, the Mamba `conv1d`, and every RMSNorm.
---

## 2. Why Low Precision

The constraint at LLM inference is not FLOPs. It's memory bandwidth.

Every generated token requires reading every weight from memory. A 30B-parameter model at BF16
stores 60 GB of weights. For a single token, with a batch of one, the GPU spends most of its time
waiting on memory — the compute units are underutilized because the data can't arrive fast enough.
This is the memory-bandwidth wall, and it's the dominant cost in autoregressive generation.

Reducing precision attacks this problem directly. Cut from 16 bits to 8 and you halve the
weight memory. Cut to 4 and you quarter it. The throughput gain is roughly proportional: 4-bit
weights can feed the compute units roughly twice as fast as 8-bit, and four times as fast as 16-bit.
Energy follows the same curve — moving bits costs picojoules; computing on them costs less.

But quantization is lossy. If you naively round BF16 weights to 4 bits without the scaffolding
this post describes, perplexity doesn't drop by a few points — it explodes. On LLaMA-7B at W4A8,
naive round-to-nearest produces a WikiText2 perplexity of 151.4, against 10.4 at full precision.
The model isn't degraded — it's broken.

The rest of this post is about the scaffolding that closes that gap: the formats that make 4-bit
representation possible, and the training methods that make the model tolerate it.

> **Sidebar: Why 4 bits isn't crazy — the biological prior**
>
> The brain is the existence proof that intelligence runs on low precision. Bartol et al. (2015,
> *eLife*) reconstructed hippocampal synapses and found approximately 26 reliably distinguishable
> strength states — log₂(26) ≈ 4.7 bits per synapse. That is startlingly close to NVFP4's
> effective 4.5 bits.
>
> But this is a shared *principle* — system-level robustness to component-level imprecision —
> not a shared *mechanism*. The differences matter:
>
> - **The bits measure different things.** Synaptic precision is an information-theoretic storage
>   bound under molecular noise. NVFP4's 4 bits is an engineered codec width chosen to feed a
>   Tensor Core datapath. They answer different questions.
> - **Synapses are stochastic; NVFP4 is deterministic.** Probabilistic vesicle release means
>   the brain computes *through* noise; quantization treats noise as error to minimize.
> - **NVFP4's dynamic range comes from explicit two-level block scaling.** Nothing in the brain
>   corresponds to "16 adjacent synapses share an exponent." Biological dynamic range is managed
>   by homeostatic scaling and divisive normalization — slow, adaptive, biochemical.
> - **DNNs accumulate dot products in FP32.** The brain has no high-precision accumulator — just
>   leaky, noisy membrane potential integration.
> - **We tolerate quantization because we trained for it** (QAT, QAD, sensitivity layers). The
>   brain tolerates imprecision through redundancy and population coding.
>
> The convergence is real: *useful computation survives crude per-element precision.* The
> implementations are unrelated. The shared lesson is that robustness is a system-level property,
> not a component-level guarantee. (Bartol et al., 2015, "Nanoconnectomic upper bound on the
> variability of synaptic plasticity," *eLife*.)

---

## 3. The Format Zoo: FP8, MX, and NVFP4

The formats are not a menu. They're a ladder — each rung solves a problem the previous one couldn't.

### Integer vs Float at Low Bit-Width

At 8 bits, integer quantization (INT8) works well: 256 uniformly spaced values cover the typical
weight distribution of a normalized neural network without much loss. But push to 4 bits and
uniform quantization gives you 16 levels. That's not enough — weight distributions have outliers
and long tails, and 16 uniform bins either clip the tails (losing the largest weights) or waste
bins on sparse extremes (losing precision in the dense center).

Floating-point formats solve this by spacing values non-uniformly: more representable numbers near
zero, fewer at the extremes. At 4 bits, that asymmetry matters more than the absolute number of
levels. A 4-bit float can cover the same dynamic range as INT4 while placing resolution where the
weight mass actually lives. The cost is more complex hardware, but Blackwell makes that cost zero
by implementing it in silicon.

### FP8: The First Rung

The H100 introduced native FP8 tensor cores, and Micikevicius et al. (2022) established the
convention that stuck: two variants, two jobs.

**E4M3** (4 exponent bits, 3 mantissa bits): weights and activations. Maximum value ~448, minimum
normal ~2⁻⁶ — about five orders of magnitude. Narrow, but weights and activations live in narrow
ranges (normalized by LayerNorm, bounded by weight decay), so three mantissa bits buy precision
where it counts. **E5M2** (5 exponent, 2 mantissa): gradients. Maximum ~57,344, minimum ~2⁻¹⁴ —
eight orders of magnitude. Gradients span orders of magnitude across layers, and you can sacrifice
precision for range.

The scaffold that makes E4M3's narrow range work is *delayed scaling*: each tensor carries an FP32
scalar `s`, updated from a sliding window of recent `amax` values, that shifts the distribution
into the representable range. Different tensors, different scales. This is the conceptual ancestor
of everything that follows.

### The 4-Bit Problem

FP8 works at 8 bits. But at 4 bits, the format math gets tight: E3M0 (3 exponent, no mantissa)
gives you only powers of two. E2M1 gives 15 distinct values — adequate range (0 to 6 in magnitude)
but almost no precision. The fix that FP8 used — one scale per tensor — isn't enough when your
elements only have 15 levels to begin with. A single scale for thousands of weights means the
blocks with small values round to zero and the blocks with large values saturate. You need finer
adaptation.

### Microscale (MX) Formats: Shared Scales

The solution, standardized as the OCP Microscaling (MX) specification by AMD, Arm, Intel, Meta,
Microsoft, NVIDIA, and Qualcomm (2023), is to share a scale across a *block* of elements rather
than the whole tensor. A block of 32 FP4 elements shares one FP8 E4M3 scale. The block is small
enough that its values are locally homogeneous (one scale works for all of them), but large enough
that the per-block scale overhead is small (8 bits of scale / 32 elements = 0.25 extra bits per
element).

The key insight: the scale has more bits than the data it scales. An FP8 E4M3 scale provides ~240
levels of adjustment for each block, which is plenty — the block's values are all within the same
order of magnitude anyway. The scale's job isn't precision; it's to place the block correctly on
the number line. The element's job isn't range; it's to provide local resolution. Together, they
split the representation problem into two parts that floating-point formats previously tried to
solve in a single number.

### NVFP4 (E2M1): Blackwell's Native Format

NVFP4 is NVIDIA's Blackwell-native implementation of the microscaling idea, with a block size of
16 instead of 32. The element format is E2M1 — 1 sign, 2 exponent, 1 mantissa — giving 15
distinct finite values. What makes it work is the scaling:

- **Block scale**: one FP8 E4M3 value per 16 consecutive weights. Computed as the block's max
  absolute value divided by 6.0 (E2M1's max magnitude), then itself quantized to FP8.
- **Tensor scale**: one FP32 scalar for the entire tensor, derived from the global max.

The reconstruction is `x_real = x_fp4 × s_block × s_tensor`. Two levels of adaptation — local
(per-16) and global (per-tensor) — give the format enough degrees of freedom to represent weight
distributions that span four or more orders of magnitude with only 15 levels.

Block size 16 is a hardware decision: it matches the Blackwell Tensor Core's fragment size, aligns
with 128-bit GPU memory transactions (16 × 4 bits = 64 bits paired with 8 bits of scale), and
keeps the scale overhead at 0.5 extra bits per element. At block size 32 you'd save 0.25 bits per
element but degrade accuracy; at block size 8 you'd improve accuracy but inflate overhead and fall
off the hardware-accelerated path.

### Why NVFP4 over INT4

Integer quantization at 4 bits gives 16 uniformly spaced values. NVFP4 gives 15 non-uniformly
spaced values with two-level block scaling. The uniform distribution of INT4 is the problem: real
weight tensors are not uniform — they have dense centers and sparse tails. INT4 either wastes
bins on the tails (reducing effective precision at the center, where most weights live) or clips
the tails (losing information in the largest weights). NVFP4 addresses this across three levels of
granularity: the E2M1 format handles per-element non-uniformity, the block scale handles local
magnitude variation, and the tensor scale handles global range. INT4 has none of these.

At inference on Blackwell, NVFP4 has another advantage: the hardware does the dequantization
inside the Tensor Core. An INT4 matmul would need a separate scale-application step. An NVFP4
matmul is a single fused operation — the block scales and tensor scale are applied during the
multiply-accumulate, not before or after it.

---

## 4. The Accuracy-Recovery Ladder: PTQ, then QAT or QAD

If the format zoo is about *how* to represent low-precision numbers, the recovery ladder is about
*how to get the model to tolerate them*. There are really two rungs. The first is **PTQ** —
calibrate and apply scales, no training. The second is **quantization-aware training**: train the
model through fake-quantization so it adapts to the noise. That second rung comes in two sibling
flavors that share the *same* training machinery and differ only in their supervision signal —
**QAT**, supervised by the task's cross-entropy on hard labels, and **QAD**, supervised by
distillation against a frozen teacher's soft labels.

### PTQ: One-Shot, No Training

Post-training quantization measures the model's weight and activation distributions on a small
calibration set, computes scales, and applies them — no gradient updates, no training loop. It's
cheap, fast, and at 8 bits it works. At 4 bits, it breaks.

Three families dominate the PTQ landscape, and they attack different parts of the problem:

**GPTQ** (Frantar et al., 2022) quantizes weights one column at a time, and after each column it
redistributes the quantization error across the remaining unquantized columns. This
error-feedback loop corrects the largest distortions before they accumulate. It works well for
weights in isolation but has no mechanism for handling activation outliers.

**AWQ** (Lin et al., 2023) observes that not all weight channels are equally important: channels
multiplying large-magnitude activations carry disproportionate weight in the output. AWQ scales
those important weight channels up before quantization and scales the corresponding activations
down, preserving the product while protecting salient channels from quantization error. It's
activation-aware, but only in a static, per-channel sense.

**SmoothQuant** (Xiao et al., 2022) tackles the single hardest problem in LLM quantization:
activation outliers that are orders of magnitude larger than the rest of the distribution. Rather
than trying to quantize around them, SmoothQuant *migrates* the quantization difficulty from
activations to weights via a per-channel smoothing factor. Activations become easier to quantize;
weights absorb the difficulty and are more tolerant of it. This is conceptually elegant — it
recognizes that weights and activations face different quantization constraints and rebalances
between them.

What PTQ shares: it measures once, applies once. No feedback loop between the three components
(weights, activations, KV cache), no adaptation to the compound error they create when combined,
and no mechanism to change the model's behavior — only its representation. At 8 bits, that's
enough. At 4 bits, the LLM-QAT paper reports the baseline: naive round-to-nearest loses ~25 points
of zero-shot accuracy on LLaMA-7B. SmoothQuant loses ~23. The gap is too large for a one-shot fix.

### QAT: Train Through the Noise

Quantization-aware training inserts fake quantization operations into the forward pass —
quantize to the target format, then immediately dequantize back — and trains the model through
them. The backward pass uses straight-through estimation (STE): the gradient treats the quantizer
as an identity function, passing through unchanged. The model sees quantization noise in the
forward pass and adapts its weights to be robust to it. The supervision signal is the ordinary one
— cross-entropy on the next-token hard labels; the fake-quant in the forward pass is the only
departure from normal training.

The cost is a training run. The benefit is that the model can *change*: a weight distribution that
collapses under round-to-nearest can shift to one that quantizes cleanly. LLM-QAT (Liu et al.,
2023) demonstrated this at scale on LLaMA models up to 30B parameters. At W4A8 — 4-bit weights,
8-bit activations — QAT recovered roughly 19 of the ~25 points lost to PTQ on LLaMA-7B (60.7 vs
41.2 for naive round-to-nearest, against a 66.2 FP16 baseline). At 30B, the gap narrowed to 1.5
points.

Two findings from LLM-QAT's ablation table matter for practitioners. First, symmetric MinMax
quantization (no clipping) beats clipping-based methods like LSQ by 13 points — clipping causes
"exceptionally high perplexity" early in training that is hard to recover from. Second, 8-bit QAT
provides essentially no benefit over 8-bit PTQ; the training premium only pays off at 4 bits.

### QAD: Swap the Signal for a Teacher

QAD runs the *same* fake-quant training as QAT — same forward quantization, same STE backward. The
one thing it changes is the supervision signal: instead of (or alongside) cross-entropy on hard
labels, the student minimizes KL divergence between its logit distribution and a frozen
full-precision teacher's. Same machinery, richer target.

Why supervise with a teacher instead of the data's own labels? Because those hard labels are noisy for this
task. If the training tokens come from sampling (as in LLM-QAT's data-free generation) or from a
narrow calibration set, the next-token target may not reflect the true distribution the model should
preserve. The teacher's soft labels — the full logit distribution — provide a richer signal that
captures the model's original behavior across the entire vocabulary. The student learns not just to
predict the right token, but to match the teacher's uncertainty, its near-misses, its
distributional shape.

LLM-QAT's distillation ablation is definitive on one point: logits-only. Adding attention-map
distillation to the loss reduces accuracy (63.1 → 61.1). Adding hidden-state distillation collapses
it (63.1 → 34.9). The teacher's internal representations are not a useful training signal for a
quantized student — the student cannot reproduce representations at the teacher's precision, and
forcing it to try creates a conflicting gradient. The teacher's value is its output distribution,
period.

### The Shape of the Ladder

| Rung | Mechanism | Cost | 8-bit result | 4-bit result |
|------|-----------|------|-------------|-------------|
| PTQ | Calibrate → apply scales, no training | Minutes | Near-lossless | ~20–25 point gap |
| QAT | Fake-quant training (STE), supervised by cross-entropy on hard labels | 1 training run | No benefit over PTQ | Closes most of the gap |
| QAD | Same fake-quant training, supervised by distillation against a teacher's soft labels | 1 training run + teacher forward passes | No benefit over PTQ | A bit more than QAT — richer signal |

The ladder is not symmetric: the big jump is getting onto quantization-aware training at all. Most
of the accuracy recovery comes from letting the model adapt to the noise — the fake-quant training
that QAT and QAD share. The choice of *signal* on top of it — hard-label cross-entropy (QAT) versus
teacher distillation (QAD) — is the smaller, second-order knob. Distillation usually wins, because
soft labels carry more information than a one-hot target, but it refines; it doesn't rescue a
broken training setup. If the fake-quant training underneath is poor, the teacher won't save it.

This case study takes the distillation path — QAD — straight from a PTQ baseline. There's no
separate QAT stage to walk first: QAT and QAD are siblings, and the project picked the
teacher-supervised one.

---

## 5. Case Study: QAD on Nemotron-3-Nano

### 5a. The Model and the Target

Nemotron-3-Nano is not a dense transformer. It's a 52-layer hybrid that interleaves three
architectural primitives: Mamba2 state-space models, Mixture-of-Experts FFNs, and standard
causal attention. Thirty billion total parameters, three billion active per token (the "A3B"
designation — 30B parameters, ~3B compute per forward pass).

The layer pattern is a 52-character string:

```
M E M E M * E M E M E M * E M E M E M * E M E M E M * E M E M E M * E M E M E M * E M E
```

| Symbol | Type | Count | Role |
|--------|------|-------|------|
| `M` | Mamba2 SSM | 23 layers | Linear-time sequence mixing via recurrent state |
| `E` | MoE FFN | 23 layers | 128 experts, top-6 routing, per-token sparse computation |
| `*` | Causal Attention | 6 layers | Full quadratic attention at roughly every 7th position |

The six attention layers act as sparse lookup anchors — Mamba's compressed state is O(N) but
lossy, and periodic full attention lets the model recover precise token-level comparisons that
the SSM state has blurred. The 128 MoE experts (FFN hidden size 1856, GLU activation, squared
ReLU) provide most of the model's capacity; the routing decision — which 6 of 128 experts handle
a given token — is a sigmoid over a learned affinity score, trained with a small load-balancing
auxiliary loss.

Key dimensions: hidden size 2688, KV channels 128, 32 query heads with 2 key-value groups (grouped-query
attention), Mamba2 state dimension 128 with 64 SSM heads. All normalization is RMSNorm. The output
head is a standard vocabulary projection — `hidden_size → vocab_size`.

Why this model for a quantization case study? Because it's a worst case for PTQ. A 30B dense
transformer at W4A8 can get within ~1.5 points of FP16 on zero-shot tasks (LLM-QAT, LLaMA-30B).
A hybrid architecture with three different computational primitives, three different weight
tensor shapes, activation distributions that differ fundamentally between SSM recurrent states
and attention softmax outputs, and per-expert routing decisions that are sensitive to small
perturbations — that's a harder quantization problem. If the recipe works here, it works
anywhere.

**Hardware target:** Blackwell B100/B200, the first GPU generation with native NVFP4 Tensor Cores.
On Blackwell, an NVFP4 matmul is a single fused hardware operation — the block scales and tensor
scale are applied inside the multiply-accumulate, not as a separate pre- or post-processing step.
The throughput argument is roughly 2× over FP8 at the same batch size (4-bit operands vs 8-bit),
with memory footprint reductions of ~3.5× for weight tensors. Whether those theoretical gains
survive contact with accuracy is what the rest of this case study is about — the recipe that
spends the precision budget where it matters (§5b).

### 5b. The Recipe

The precision map is asymmetric by design. Not all layers are equal under quantization, and the
recipe reflects that — a deliberate set of choices about where to spend the precision budget.

**Weights: NVFP4 (E2M1) with two-level block scaling.** Each weight tensor is quantized to 4-bit
E2M1 — 1 sign, 2 exponent, 1 mantissa bit — giving 15 distinct values: {0, ±0.5, ±1, ±1.5, ±2,
±3, ±4, ±6}. Fifteen values is not much. What makes it work is the scaling: every block of 16
consecutive weights shares one FP8 E4M3 scale factor, and the whole tensor carries one FP32 global
scale:

```
x_real = x_fp4 × s_block × s_tensor
```

where `s_block` = `amax_block / (6.0 × s_tensor)`, quantized to FP8 E4M3. `s_tensor` is a single
FP32 scalar derived from the tensor's overall max absolute value. Block size 16 is not arbitrary:
it matches the Blackwell Tensor Core's native fragment size, aligns with 128-bit GPU memory
transactions (16 × 4 bits = 64 bits = 8 bytes, paired with the 1-byte FP8 scale), and keeps the
scale overhead to 0.5 bits per element — 4.5 bits effective. A block size of 32 would cut
overhead to 0.25 bits but degrades accuracy; a block size of 8 would improve accuracy but inflates
overhead to 1 bit per element and falls off the hardware-accelerated path.

**Activations: FP8 (E4M3).** Weights go to 4-bit; activations stay at 8-bit. This is not a
compromise made lightly — both LLM-QAT (Liu et al., 2023) and our own experiments confirm that
4-bit activations remain an open problem. FP8 E4M3 provides ~240 distinct values across ~5 orders
of magnitude, which is sufficient for per-token activation distributions when combined with
delayed scaling (tracking amax over a sliding window of recent steps).

**KV cache: FP8.** The KV cache is the dominant memory cost at long context, and it gets its own
quantization choice: FP8, not NVFP4. This is a deliberate "2× now, safely" decision — FP8 roughly
halves the cache footprint versus BF16 with far less accuracy risk than pushing the cache itself to
4 bits. Going to 4-bit KV (as LLM-QAT does, for a 4× reduction) is left on the table as future work;
see §7.

**Teacher: BF16, frozen, in eval mode.** The teacher is the unquantized pretrained Nemotron-3-Nano.
It is never updated during distillation, which means the distillation loss (KL divergence between
teacher and student logits) is equivalent to cross-entropy — the teacher's entropy term is
constant, so the gradient is identical either way.

**Sensitivity layers — kept out of NVFP4.** A central finding of this project: not all layers carry
the same accuracy budget. The shipped quantization config excludes the following from NVFP4, keeping
them at the model's higher-precision compute path:

| Component | Precision | Rationale |
|-----------|-----------|-----------|
| Embeddings | BF16 | First touch on token identity (a lookup, never a quantized GEMM); error here propagates through every subsequent layer |
| LM head | BF16 | Last touch before the vocabulary projection; directly shapes the output distribution the distillation loss compares against |
| All Mamba `conv1d` | BF16 | The short depthwise convolution is tiny in parameter count but numerically delicate — it shapes the SSM input, and quantizing it risks state corruption for negligible memory savings |
| All 6 attention layers (`q/k/v/o_proj`), plus the `in_proj`/`out_proj` of the Mamba layer feeding each | BF16 | The attention layers are scarce, load-bearing routing anchors in a mostly-Mamba stack; the config protects *all six* (not just the early ones), along with the Mamba projections immediately upstream |
| All RMSNorms | FP32 | Normalization statistics are tiny vectors where even single-bit errors shift the mean and variance of every token that passes through |

Everything else — all MoE expert FFNs (**including the routers**, which the config leaves quantized),
all remaining Mamba SSM projections — runs at NVFP4 weights with FP8 activations. The router choice is
worth flagging: a common instinct is that 128-expert top-6 routing is too sensitive to quantize, but
the shipped recipe quantizes it anyway. Whether that's safe is a natural ablation (see §7).

**Loss: KL(teacher ‖ student) with temperature.** The student minimizes
`KL(p_teacher ‖ p_student)` on the logit distribution, with a temperature parameter T that
softens both distributions. Additional hard-label cross-entropy on the calibration tokens provides
a ground-truth anchor. The loss is normalized per valid token (ignoring padding).

**Training.** Optimizer: AdamW with cosine decay. Calibration data drawn from a curated mixture
covering the target deployment domain. The student is initialized from the pretrained weights
(not random, not PTQ-calibrated) — same initialization as the teacher, which means the only
difference at step zero between teacher and student is the quantization noise.

---

## 6. Training in FP4: The Harder Half of the Pipeline

Everything so far quantizes a *finished* model. PTQ, QAT, QAD all take a network trained in high
precision and compress it for inference; the backward pass never enters the picture. The real
frontier is training *in* FP4 — running the gradient computation itself through 4-bit GEMMs. That is
a different, harder problem, and 2025–2026 produced the first results that make it real.

### The scaling spine: FP32 → FP16 → FP8 → FP4

Low-precision training has climbed a ladder for a decade, and every rung has the same shape: quantize
the bulk matmul, protect the accumulation.

- **Mixed precision (FP16).** Micikevicius et al. (2018) ran the forward and backward GEMMs in FP16
  but kept an FP32 *master copy* of the weights, accumulated in FP32, and used loss scaling to keep
  small gradients from flushing to zero. One global scale for the whole network.
- **FP8.** The Hopper generation moved the GEMMs to 8-bit with a *per-tensor* scale tracked over a
  sliding window of recent amax values (delayed scaling, §3). Finer adaptation, same invariant: FP8
  operands, FP32 accumulator.
- **FP4.** The microscaling formats (§3) push to a *per-block* scale — 16 or 32 elements share one
  scale — which is what makes 4-bit operands viable at all.

The invariant never changes: master weights, accumulator, and optimizer state stay in FP32. What
drops is the precision of the operands feeding the matmul.

### Why the backward pass is where it breaks

A training step has three matmuls, not one: the forward pass (**Fprop**), the activation gradient
(**Dgrad**), and the weight gradient (**Wgrad**). Under FP4 they are not equal. Quantizing Fprop and
Dgrad costs only a modest amount — on the order of ~10% more tokens to reach the same loss,
recoverable just by training a little longer. Quantizing **Wgrad** is what *destabilizes* training:
in Cim et al.'s staged MXFP4 ablation, adding Wgrad jumps the token overhead from ~10% to ~26% and
the loss curve begins to diverge.

The reason is structural. The weight gradient is a sum over the batch-and-sequence dimension — every
token contributes to it — so a *structured* quantization error in Wgrad writes directly into the
parameter-update direction, where it steers the optimizer itself off course. An error in Fprop or
Dgrad only perturbs a single forward or backward signal; an error in Wgrad biases *where the model
moves next*. That is why the backward weight-gradient path, not the forward pass, is where 4-bit
training actually breaks.

### The fix, and a contradiction worth understanding

The fixes are specific, and they are applied surgically — each tool to the one tensor whose failure
mode it addresses:

- **Hadamard rotation** on the Wgrad inputs. A Hadamard transform is an orthogonal rotation; rotate
  both GEMM operands and, because it is its own inverse (`H Hᵀ = I`), the rotations cancel in the
  product and the math is unchanged — but the *quantization* now happens on rotated operands, where
  concentrated outliers have been smeared into a more Gaussian, more block-scale-friendly distribution.
- **Stochastic rounding** on the gradient tensors only. SR makes rounding unbiased, which matters most
  where bias does the most damage — the gradients. Applied to weights or activations it *adds*
  variance and diverges.
- **2D block scaling** on the weights, so the forward and backward passes quantize each weight
  *identically*. If they don't, backprop computes the gradient of a slightly different function than
  the forward pass evaluated — a chain-rule violation.

Here the two leading results appear to contradict each other, and the contradiction is the most
instructive part. Cim et al. (2026), training MXFP4 on AMD MI355X, found that **deterministic**
Hadamard rotations restore stability while **random** Hadamard *and* stochastic rounding both fail.
NVIDIA (2025), pretraining a 12B hybrid Mamba-Transformer on 10T tokens in NVFP4, used **random**
Hadamard and stochastic rounding and *succeeded*, matching the FP8 baseline.

The resolution isn't that one of them is wrong. It is **selectivity plus format headroom.** Both
papers agree that applying these tricks blindly to the whole pipeline fails — NVIDIA's own ablation
shows SR on weights and activations diverging, exactly Cim's warning. NVIDIA wins by applying each
tool only to the tensor that needs it (SR → gradients, Hadamard → Wgrad), on a *finer* format:
NVFP4's block-16 with an E4M3 scale carries more precision per block than MXFP4's block-32 with a
power-of-two E8M0 scale, and that extra headroom is enough to absorb the variance random methods
inject. On Cim's coarser format there was no headroom to spare, so the structure had to come from a
*deterministic* transform instead. The right question was never "random or deterministic Hadamard" —
it is "which tensor, which format."

### Precision can vary across time, not just across layers

One more idea from the NVFP4 pretraining run is worth carrying back. The 4-bit loss tracked FP8 to
within ~1% for most of training, then the gap widened to ~1.5% during the learning-rate *decay*
phase. Switching just that phase back up to BF16/MXFP8 — at 8.2T of 10T tokens — closed it to ~0.5%
for about 6% overhead. Precision isn't an all-or-nothing choice made once: the sensitive window can
be a *phase of training*, not just a *set of layers*. The case study's sensitivity map (§5b) protects
layers; this protects a moment in time. Same instinct, different axis.

The throughline back to this post: the NVFP4 pretraining result is on the *same architecture family*
as the case-study model — a hybrid Mamba-Transformer — using the *same format* this recipe deploys
for inference. Inference quantization and training quantization are converging on the same model.
Extending the case-study recipe from FP4 inference to FP4 gradients — quantizing the backward pass,
paying the price of admission in Hadamard-rotated, stochastically-rounded gradients — is the most
direct next step there is.

---

## 7. What's Still Open

Training in FP4 (§6) is the largest open frontier. The inference recipe itself leaves several smaller
threads open too.

**4-bit KV cache.** The current recipe quantizes the KV cache to FP8 — a safe 2× reduction. LLM-QAT
shows 4-bit KV is viable for a 4× reduction (24 GB → 6 GB at 32k sequence length on a 30B *dense*
model) at a few points of accuracy. But those dense numbers overstate the lever here: Nemotron-3-Nano
is a hybrid where only 6 of 52 layers hold a KV cache at all (the 23 Mamba layers keep an O(1)
recurrent state; the MoE layers hold none), so its KV footprint is more than 10× smaller than a dense
transformer's. Weights dominate memory until very long context — KV compression only starts to pay
past ~128K tokens.

A more aggressive option than scalar FP8/INT4 is *vector* quantization of the cache. **TurboQuant**
(Zandieh et al., 2025) is a training-free, data-oblivious quantizer — a fixed random rotation plus a
Lloyd-Max codebook tuned for the resulting near-Gaussian coordinates — that lands within a small
constant factor of the information-theoretic distortion bound at any bit-width, with no calibration.
Its headline is 4× quality-neutral; in practice, independent implementations find the safe operating
point closer to ~2×, with a full-precision window over recent tokens and asymmetric key/value bit
budgets (keys and values carry very different norm scales). Stacking it on NVFP4 weights is the
natural composite recipe, and the error math is encouraging: KV error passes through softmax, where
it dominates the sub-percent weight error several-fold — so the NVFP4 weights are nearly free in the
stack, and the open question reduces to whether vector-quantized KV is acceptable on this hybrid at
all. With only six attention layers for the error to propagate through, it may be more forgiving than
a pure transformer. Untested here, but a cheap and concrete ablation.

**Should the routers be quantized at all?** The shipped recipe quantizes the 128-expert top-6 routers
to NVFP4 along with everything else. Routing is a discrete, sensitive decision — a small perturbation to
an affinity score can send a token to the wrong expert. Whether 4-bit routing is safe, or whether the
routers deserve a place on the sensitivity list, is an open ablation this project hasn't run.

**4-bit activations.** Both LLM-QAT (2023) and this work keep activations at FP8. 4-bit activations
remain a standing hard problem — the dynamic range and outlier structure of activations resist the
block-scaling tricks that work for weights. This is the honest limitation of every "4-bit LLM" result,
including this one: it's W4A8, not W4A4.

**Per-expert, long-context, and sparsity.** Three smaller open threads. (1) *Per-expert quantization*:
the 128 experts have very different activation rates; the heavily-used experts might deserve more
precision than the rarely-routed ones, rather than one recipe for all. (2) *Long-context block-scale
stability*: whether the per-16 block scales remain well-behaved far beyond the calibration sequence
length is untested here. (3) *Stacking with structured sparsity*: Blackwell accelerates 2:4 sparsity
independently of NVFP4 — whether the two compose without compounding accuracy loss is unknown.

---

## 8. References

**Formats**

- Micikevicius et al. (2022). *FP8 Formats for Deep Learning.* [arXiv:2209.05433](https://arxiv.org/abs/2209.05433)
- Rouhani et al. (2023). *Microscaling Data Formats for Deep Learning.* [arXiv:2310.10537](https://arxiv.org/abs/2310.10537) — with the OCP Microscaling (MX) Specification v1.0 (AMD, Arm, Intel, Meta, Microsoft, NVIDIA, Qualcomm, 2023).
- NVIDIA (2025). *Introducing NVFP4 for Efficient and Accurate Low-Precision Inference.* [NVIDIA Technical Blog](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)

**The accuracy-recovery ladder (PTQ, QAT, QAD)**

- Frantar, Ashkboos, Hoefler, Alistarh (2023). *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.* [arXiv:2210.17323](https://arxiv.org/abs/2210.17323)
- Lin et al. (2023). *AWQ: Activation-Aware Weight Quantization for LLM Compression and Acceleration.* [arXiv:2306.00978](https://arxiv.org/abs/2306.00978)
- Xiao et al. (2023). *SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models.* [arXiv:2211.10438](https://arxiv.org/abs/2211.10438)
- Liu et al. (2023). *LLM-QAT: Data-Free Quantization-Aware Training for Large Language Models.* [arXiv:2305.17888](https://arxiv.org/abs/2305.17888)

**Low-precision training (§6 frontier)**

- Micikevicius et al. (2018). *Mixed Precision Training.* [arXiv:1710.03740](https://arxiv.org/abs/1710.03740)
- NVIDIA. *Using FP8 with Transformer Engine* (FP8 primer). [Transformer Engine documentation](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)
- Cim et al. (2026). *Pretraining Large Language Models with MXFP4 on Native FP4 Hardware* (full-pipeline MXFP4 training on AMD MI355X). [arXiv:2605.09825](https://arxiv.org/abs/2605.09825)
- NVIDIA (2025). *Pretraining Large Language Models with NVFP4.* [arXiv:2509.25149](https://arxiv.org/abs/2509.25149)

**KV cache quantization (§7 frontier)**

- Zandieh, Daliri, Hadian, Mirrokni (2025). *Online Vector Quantization with Near-Optimal Distortion Rate* (TurboQuant). [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)

**Background**

- Bartol et al. (2015). *Nanoconnectomic upper bound on the variability of synaptic plasticity.* eLife. (§2 biological-prior sidebar)
