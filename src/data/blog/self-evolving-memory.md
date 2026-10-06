---
title: "Self-Evolving Memory for LLM Agents: A Hands-On Study"
pubDatetime: 2026-10-06T12:00:00Z
description: "Five memory systems and a full-context baseline on LoCoMo, LongMemEval and BEAM, then eight memory configurations on task streams: the scorer decides the headline, full context is hard to beat until it stops fitting, and self-evolving memory did not learn."
tags:
  - agent-memory
  - llm-agents
  - benchmarks
  - evaluation
  - self-evolving
draft: false
---

I evaluated five agent memory systems and a full context baseline across three long term memory benchmarks, then tested eight memory configurations on task streams where memory is supposed to improve with experience. The study used two backbones (an open 27B model and Claude Sonnet 5), one judge, one retrieval budget, and complete token accounting. Five key findings emerged.

**1. The metric can flip the conclusion.** On LoCoMo, the official F1 ranks the open model above Sonnet on four of the five memory systems (the fifth is a tie), while the LLM judge ranks Sonnet above the open model on all five. Against 98 blind human labels, the judge aligns more closely with human judgments (κ = 0.76 vs. 0.60), suggesting that the choice of scorer can materially change the apparent winner.

**2. If the history fits, full context is hard to beat, and it is not always the expensive option.** Below about 115K tokens no memory system beats reading everything on LoCoMo, LongMemEval-S or BEAM-100K. Full context pays for the history on every question; LLM-written memory pays once, at ingestion, by reading the history 4.5 to 9 times over. Which is cheaper depends on how many questions a history gets and whether its prefix stays cached.

**3. What a memory system stores determines what it remembers well.** Systems that extract structured facts perform best on dates, preferences, and aggregation across sessions, but often lose information about what the assistant itself said. Systems that retain raw conversation text show the opposite pattern.

**4. Once the history outgrows the context window, LLM-written memory starts to win.** At 1M tokens, the full-context baseline must truncate the beginning of the conversation, and its BEAM score drops from 0.47 to 0.32. The two LLM-written memory systems largely maintain their performance.

**5. Self-evolving memory did not become more accurate with experience; the only improvement over a stream was efficiency.** When memory improved accuracy, the gain was static: it was already present from the first question, persisted when Dynamic Cheatsheet's final sheet was frozen from the start, and transferred just as well from a sheet written on a different subject. Accuracy curves remain flat for every system on both backbones, while retrieving raw past attempts is most often best or tied for best. On a coding stream, however, ACE's reduction in turns became larger in the second half of the stream than in the first.

## 1. Why agents need memory

An LLM agent's context window is its only native working memory at inference time. Once a task ends, the model does not carry that task-specific state forward on its own. Anything that makes an agent useful over weeks therefore has to be stored outside the model and brought back when needed: what the user said last month, which fix worked on the previous ticket, which approach failed. "Agent memory" is the name for that machinery.

Two different jobs share the name. The first is **remembering**: given a long history, answer a question about it. The history is fixed; the question is whether the right piece of it reaches the prompt. The second is **learning**: given a stream of tasks, do later tasks go better because of earlier ones? Here, memory is supposed to change over time, turning past attempts into reusable knowledge. Papers often call this second job self-evolving memory, and it is the one associated with the stronger claims.

This post asks two questions using one harness. Part A (§4) asks: which memory abilities do current systems deliver on standard benchmarks, at what cost, and does the answer depend on the backbone? Part B (§5) asks: when memory is updated after every task using pass/fail feedback, does performance improve over the stream? And if it does, is that learning—or simply good advice accumulating in the prompt?

I did not build a new memory system. The study uses existing systems, public benchmarks, and controls.

## 2. What "self-evolving" actually means

### 2.1 Prompt-driven memory consolidation

These methods use LLM prompting to transform task experiences into reusable external
memory without updating model weights. Their memory contents span **episodic memory** —
records of specific tasks, actions, and outcomes — and **semantic or procedural memory** —
generalized facts, strategies, workflows, and code.

| Method | What is stored? | How is memory consolidated? |
|---|---|---|
| **[Dynamic Cheatsheet](https://arxiv.org/abs/2504.07952) (DC-CU)** | A cheatsheet containing heuristics, formulas, solution sketches, and reusable code; primarily semantic and procedural knowledge | A curator rewrites the cheatsheet using the existing memory and the latest generated solution, potentially merging, revising, or removing content |
| **Dynamic Cheatsheet (DC-RS)** | The cheatsheet plus historical query–generated-answer pairs; combines episodic records with distilled knowledge | Retrieves similar past examples and synthesizes an updated cheatsheet before answering the current query |
| **[ACE](https://arxiv.org/abs/2510.04618)** | A structured playbook of domain knowledge, tool-use rules, procedures, code, and failure patterns, with unique IDs and helpful/harmful counters | A reflector extracts lessons, a curator proposes incremental additions, and programmatic merging and semantic deduplication maintain the playbook |
| **[ReasoningBank](https://arxiv.org/abs/2509.25140)** | Task queries, original trajectories, and associated memory items structured as `{title, description, content}` | Distills strategies and pitfalls from self-judged successful and failed trajectories, then appends the resulting items without additional pruning in the basic implementation |

ReasoningBank retains episodic traces in storage, but the agent primarily receives
**distilled memory items** at inference time. Its implementation retrieves similar
historical queries and injects their associated strategies into the system prompt.

The methods differ in their supervision and model requirements:

| Method | Ground-truth labels | Judge or reflection model | Embedding model |
|---|---|---|---|
| **DC-CU** | Not required in the original method | A curator assesses usefulness and potential errors; no separate success/failure classifier is required | Not required |
| **DC-RS** | Not required; stored answers are model-generated | A curator performs assessment and synthesis | Used for historical-query retrieval (`text-embedding-3-small` in the reference implementation) |
| **ACE** | Supports both labeled and unlabeled adaptation; execution feedback can guide adaptation | Reflector and curator roles | Used for semantic deduplication; incremental merging itself does not require embeddings |
| **ReasoningBank** | No ground-truth reference is required for memory induction | An LLM judge produces success/failure pseudo-labels, followed by an extraction step | Used for similarity-based retrieval (`gemini-embedding-001`) |

These roles do not require separate model families: the papers use the same backbone
model for generation and memory-processing roles in their principal configurations.

**"No ground-truth labels" should not be equated with "no supervision."** ReasoningBank
relies on model-generated judgments. In ACE's publicly released generic adaptation code,
disabling ground-truth answer text still leaves correctness feedback computed against the
target answer. This observation applies to that implementation path and should not be
generalized to every reported experiment.

The sequence **DC → ACE → ReasoningBank** is best understood as a comparison of
mechanisms, rather than a chronological progression. Their initial publication order was
DC (April 2025) → ReasoningBank (September 2025) → ACE (October 2025). Nor do they form a
strict progression from episodic to semantic to procedural memory: all three distill
reusable knowledge, while differing in how they preserve, update, and retrieve it.

### 2.2 Learned memory policies

Memory management can also be learned through parameter optimization, with reinforcement learning assigning credit to memory decisions based on downstream task outcomes. [Memory-R1](https://arxiv.org/abs/2508.19828) trains a Memory Manager to choose among explicit operations — ADD, UPDATE, DELETE, and NOOP — over an external memory bank, while an Answer Agent learns to retrieve and use relevant memories. [MemAct](https://arxiv.org/abs/2510.12635) incorporates memory editing directly into the agent's action space, allowing it to prune selected interaction turns and write compact summaries during task execution; its objective rewards task success while penalizing resource-limit violations. [Mem-α](https://arxiv.org/abs/2509.25911) frames memory construction as a sequential decision problem, training an agent to populate and update core, episodic, and semantic memory components using downstream question-answering accuracy as feedback. [MemSearcher](https://arxiv.org/abs/2511.02805) operates at the trajectory level, maintaining a compact working-memory state while jointly optimizing reasoning, search, and memory updates through multi-context GRPO, which propagates trajectory-level advantages across successive contexts.

In all of these systems, the memory-management policy is learned in model parameters, while the memory contents remain external or in working context. Learning the policy during training therefore does not imply that model weights continue to update during deployment.

### 2.3 Meta-evolution

[MemEvolve](https://arxiv.org/abs/2512.18746) treats the memory mechanism itself as something that can evolve. In its inner loop, an agent accumulates task experience under a candidate memory system. In the outer loop, execution feedback is used to revise the code governing memory encoding, storage, retrieval, and management. Candidate architectures are then evaluated on task performance, token usage, and latency. In this way, adaptation extends beyond updating stored knowledge to redesigning how experience is retained and reused.

A broader caution comes from [EvoAgentBench](https://arxiv.org/abs/2607.05202), which reports that “no current automatic method sustains positive gain in all settings.” In the authors’ words, “every automatic method still exhibits negative transfer in at least one scaffold–backbone–domain setting.” Importantly, this result comes from evaluations of [Memento](https://arxiv.org/abs/2508.16153), [ReasoningBank](https://arxiv.org/abs/2509.25140), and [GEPA](https://arxiv.org/abs/2507.19457) across two agent scaffolds and three backbone models; it is not a direct evaluation of MemEvolve. The broader lesson is that agent evolution should be assessed for both gains and regressions across deployment settings, rather than assuming that each successive evolution step will reliably improve performance.

### 2.4 Two questions the studies leave open

Across these approaches, two evaluation questions remain difficult to separate.

The first is **memory consolidation versus simple storage**: does the agent transform experience into reusable knowledge, or does it improve simply because it can retrieve more of its history? [EvoMemBench](https://arxiv.org/abs/2605.18421)’s Finding 6 identifies the formation of reusable knowledge as a key bottleneck.

The second is **similarity-driven reuse versus generalizable learning**: do gains extend to genuinely different tasks, or mainly to new tasks that resemble previous ones? [Dynamic Cheatsheet §4.6](https://arxiv.org/html/2504.07952#S4.SS6) shows benefits from recurring task structure and example ordering, while [Evo-Memory](https://arxiv.org/abs/2511.20857)’s RQ2 reports a strong correlation between memory gains and within-dataset task similarity.

Part B is built around these two questions. For the first, ExpRAG serves as the matched baseline: it stores and retrieves past attempts without transforming them, under the same context budget. For the second, two controls test whether gains depend on similarity to earlier tasks: Dynamic Cheatsheet's final sheet frozen from the start, and a sheet written on a different subject.

## 3. How memory is benchmarked

Three public benchmarks are commonly used to test whether an agent can remember over long horizons: [LoCoMo](https://arxiv.org/abs/2402.17753) (long two-person conversations), [LongMemEval-S](https://arxiv.org/abs/2410.10813) (500 questions, each with roughly 115K tokens of user–assistant history), and [BEAM](https://arxiv.org/abs/2510.27246) (synthetic conversations from 100K to 10M tokens, scored on 10 memory abilities). Although all three test long-term memory, they differ substantially in how performance is scored—and that distinction is essential before interpreting any leaderboard.

**The same answers can produce very different scores.** LoCoMo ships with an official scorer: token-level F1 for most categories and a substring-based test for the adversarial category. Mem0’s published 92.5, however, comes from a different evaluation setup—an LLM judge using a relatively permissive rubric, where one correct item can be sufficient and dates may differ by up to 14 days.

I ran both scorers on exactly the same answers: all ten conversations, full context in the prompt, with Sonnet generating the responses. On categories 1–4, the official F1 yields 0.49, while the LLM judge gives 0.92.

The two scorers also disagree about which backbone is better. On four of the five systems the F1 ranks the open 27B model above Sonnet (on the fifth, Hindsight, they tie), while the judge ranks Sonnet above the open model on every system. §6 checks both against human labels.

![Figure 1. LoCoMo categories 1–4, judge and official F1, per system and backbone.](/images/agentmemory/f1_locomo_scorers.png)

**What does “abstention” actually measure?** In LoCoMo, abstention changes with how much information is shown to the model; on BEAM, it can even change depending on which model writes the memory. Abstention is therefore not just a property of the backbone—it also depends on the memory and prompt.

**Graded judges are noisier than binary ones.** Re-running LoCoMo’s yes/no judge barely changes the result, while BEAM’s partial-credit judge can shift some ability scores by as much as 0.13. I therefore average BEAM results across multiple conversations rather than relying on individual runs.

For the comparisons below, I use the same judge model across systems, report official scores where available, and validate the judge against blind human labels from both backbones.

## 4. Part A — agent memory itself

### 4.0 What I expected

I recorded these four predictions, after the one-conversation pilots and before the full-scale runs. The results in §4.2 bear on each of them; Part B introduces a fifth.

1. **Each memory system has its own strengths.** No system outperforms full context across every ability. Systems that extract structured facts should do best on dates and contradictions; systems that preserve raw text should do best on fine-grained recall; and retrieval-based systems should fall behind full context when answering requires integrating evidence across sessions.
2. **Stronger models need memory less.** The benefit of adding memory should shrink as the backbone model becomes stronger.
3. **Who writes the memory matters.** The same memory system should perform differently depending on which model generates the memory.
4. **Scorers agree on systems, not on models.** Changing the scorer should preserve the ranking of memory systems while potentially reversing the ranking of backbone models.

### 4.1 How I ran it

**Five systems, one answering prompt.** Full context puts the entire conversation history into the prompt; it is the standard long context baseline provided by each benchmark. BM25 is plain keyword search over conversation turns. [Mem0](https://arxiv.org/abs/2504.19413) extracts facts from the history into a memory store, updating or deleting old ones as new information arrives. [AMEM](https://arxiv.org/abs/2502.12110) writes a note for each turn, tags it, and links it to related notes. [Hindsight](https://github.com/vectorize-io/hindsight) runs as a server that stores the history and retrieves relevant information on request. For every system except full context, retrieved memory is capped at 4,096 tokens and inserted into the benchmark's official answering prompt without otherwise changing it. The only thing that changes across systems is what the model gets to read.

**Two backbones.** An open 27B model (Qwen3.8-27B, served locally) and Claude Sonnet 5, both without extended thinking. The same model that answers also writes the memory; there is no stronger hidden model doing the extraction.

**One judge.** Claude Sonnet 5 scores every answer using each benchmark's own judge template, and Mem0's rubric for LoCoMo. Where a benchmark provides an official scorer, I report that too. Abstention gets two numbers: how often the model correctly says "not in the history," and how often it says so when the answer is actually there.

**How much ran.** On the open model, I ran everything: all ten LoCoMo conversations, all 500 LongMemEval questions (with a 150-question slice for the two most expensive systems), 20 BEAM conversations at 100K tokens, and five at 1M. AMEM ran on Sonnet for LoCoMo only. Sonnet ran smaller slices of the same items to test whether the same qualitative patterns appeared across backbones, rather than to produce a standalone ranking of systems. Every number below comes from a single run.

**Every token counted.** All model calls for answering, writing memory, and judging go through the same logging client. Each result therefore includes the full model-call cost, including the upfront cost of building the memory.

### 4.2 What happened

**The short version.** Up to about 115K tokens of history, reading everything is the best or tied-best choice on every benchmark, on both backbones. Memory earns its place in three narrower situations: on specific abilities where extracted facts beat the raw transcript (dates, preferences, contradictions); when the history no longer fits the window (BEAM at 1M); and on cost, where the memory systems reach most of full context's accuracy from a small fraction of the tokens per question.

**LoCoMo.**

| judge, categories 1–4 | full | bm25 | mem0 | hindsight | amem |
|---|---|---|---|---|---|
| Sonnet | **0.915** | 0.840 | 0.827 | 0.897 | 0.717 |
| Qwen3.8-27B | **0.860** | 0.765 | 0.783 | 0.826 | 0.664 |
| tokens read per question | ~30K | ~4K | ~0.5K | ~5K | ~4K |

On the judge, the order is the same on both backbones: full context, then Hindsight, then BM25 and Mem0 close together, then AMEM. Mem0 gets within 0.1 of full context while reading about one-fiftieth as many tokens. On the official F1 the picture changes: the backbones swap places, and on Sonnet full context drops from first to third, behind Hindsight and Mem0. §6 explains why the judge is the one to trust here.

One pattern is worth taking from the F1 anyway. Mem0 wins the temporal category by a wide margin on F1 but only ties on the judge. Extracted facts carry explicit dates, so Mem0's answers *contain* the date more often; it is not better at temporal reasoning.

**LongMemEval.**

| judge accuracy, Qwen | full (500) | bm25 (500) | mem0 (500) | hindsight (150) | amem (150) |
|---|---|---|---|---|---|
| task average | **0.782** | 0.666 | 0.678 | 0.777 | 0.633 |
| what the assistant said | 1.000 | 0.911 | **0.661** | **0.176** | 1.000 |
| user preference | 0.367 | 0.367 | 0.533 | **0.889** | 0.556 |
| multi-session | 0.692 | 0.376 | 0.684 | **0.872** | 0.436 |
| temporal | 0.805 | 0.609 | 0.571 | **0.850** | 0.550 |
| knowledge update | 0.846 | 0.821 | 0.731 | **0.875** | 0.542 |
| false abstention | 0.015 | **0.202** | 0.128 | 0.085 | 0.241 |

![Figure 2. LongMemEval-S per question type, Qwen, all five systems.](/images/agentmemory/f2_longmemeval_types.png)

The task average says full context and Hindsight tie, with the rest well behind. The rows say more.

- **Fact extraction loses some of what the assistant said.** Mem0 and Hindsight mainly write down facts about the user and the world. Ask what the assistant itself said earlier, and their scores drop to 0.66 and 0.18, while the two systems that preserve raw text score 1.0.
- **That same extraction appears to help on the harder question types.** Hindsight beats full context on multi-session, temporal, knowledge-update, and preference questions, using a 4K memory block instead of a 115K history.
- **BM25 says "not in the history" for a fifth of the questions that are.** Its keyword retrieval misses the relevant evidence, and the model honestly reports that it cannot find the answer. Accuracy alone would hide this failure mode, which is why I report two abstention numbers.

Two caveats. Hindsight and AMEM ran on a 150-question slice, the other three on all 500, so don't read much into small gaps between the two groups. And Sonnet's slices are small (20 to 100 questions): its full-context 0.80 matches the open model's 0.78, which is all I use them for.

**BEAM: the crossover.**

| 10-ability average, Qwen | 100K (20 conversations) | 1M (5 conversations) |
|---|---|---|
| full context | **0.471** | 0.324 (cut to fit the window) |
| hindsight | 0.425 | **0.452** |
| mem0 | 0.421 | 0.427 |
| bm25 | 0.332 | 0.330 |
| amem | 0.386 | — |

![Figure 3. BEAM 10-ability average at 100K and 1M tokens, Qwen.](/images/agentmemory/f3_beam_crossover.png)

At 100K, the same ranking holds and full context wins. At 1M, the conversation no longer fits in the context window. Under the benchmark's official truncation rule, only the most recent part is kept, so the beginning disappears. Knowledge-update accuracy falls to zero, and full context drops to roughly BM25's level. The two LLM-written memory systems, by contrast, retain about their 100K performance. This is the one regime in the study where memory clearly beats reading everything, and exactly the kind of regime memory systems are meant to handle.

**Who writes the memory matters.** On BEAM at 100K, Mem0 written by Sonnet wins contradiction resolution outright. The same Mem0 pipeline, with the same code and answering prompt but memory written by the open model, scores near zero on the same category (0.07 versus 0.42). The memory writer is therefore part of the system: changing it can change what the memory system is good at.

![Figure 7. Accuracy against input tokens per question (ingestion not included).](/images/agentmemory/f7_cost.png)

**Cost.** Answering is cheap for every memory system; building the memory is not. Mem0 processes a LongMemEval history about 4.5 times over to extract its facts, and AMEM about nine times. In dollar terms on Sonnet, full context costs about \$0.54 per LongMemEval question because each question has its own history, leaving nothing to amortize across questions. Mem0 costs about \$2.47 to ingest a history, but answering from that memory costs only a fraction of a cent per question. If several questions share the same history, that upfront cost pays for itself after about five questions—or about thirty-five if the full-context prefix stays cached. On LoCoMo, where many questions do share one conversation, cached full context costs less than a cent per question, making it cheaper than BM25.

## 5. Part B — self-evolving memory

### 5.0 What I expected

**Expectation 5: only the memories that write distilled lessons (DC and ACE) should improve over the stream; retrieval memories should stay flat.** Expectation 2 from Part A applies here too: memory should help the weaker backbone more.

### 5.1 How I ran it

**The stream.** Part B follows the Evo-Memory protocol ([arXiv 2511.20857](https://arxiv.org/abs/2511.20857)) on my own runner, since no official code was available. Questions arrive one at a time in a fixed, seeded order. For each one, the memory system hands the model a block of at most 4,096 tokens, the model answers, an exact-match scorer says correct or wrong, and the memory system updates itself from the question, the model's answer and that one bit. The reference answer never reaches the memory. Streams: three MMLU-Pro subjects (Engineering, Economics, Philosophy; 500–970 questions each), GPQA-Diamond (198) and AIME 2024 and 2025 (30 each). The open model ran three orders of each MMLU-Pro stream and one of the others, with a 16K-token thinking budget; Sonnet ran one order of Philosophy, GPQA and AIME without thinking.

**Eight systems, one interface.** Every system sees the same questions in the same order, gets the same feedback, and answers through the same prompt. They differ in what they keep and how they update it.

| System | What it keeps | How it updates |
|---|---|---|
| **none** | Nothing | — |
| **ExpRAG** | Raw records: question, the model's answer, correct or wrong | Appends one record per question; retrieves the most similar past records by embedding. No LLM touches the memory. |
| **DC** (Dynamic Cheatsheet) | One cheatsheet of strategies, formulas and pitfalls | After each answer, a curator LLM rewrites the whole sheet, seeing the correctness bit |
| **dc-frozen** | DC's *final* sheet from a completed run | Injected from the first question and never updated |
| **ACE** | A playbook of bullets with helpful/harmful counters | A reflector extracts lessons, a curator makes small edits; the worst bullets are dropped at the cap |
| **Mem0** | Extracted facts | LLM extraction, then add / update / delete against similar stored facts |
| **AMEM** | One linked note per question | Builds a note, links it to related notes, revises the neighbours |
| **Hindsight** | The Part A system, in recall mode | Retains one record per question (question, answer given, correct or wrong); recalls the most relevant facts before the next one |

ExpRAG and DC are my implementations; ACE, Mem0, AMEM and Hindsight run their published code behind the common interface. Details for the first four are in [Evo-Memory §3.2](https://arxiv.org/html/2511.20857#S3.SS2), [Dynamic Cheatsheet §2.1](https://arxiv.org/html/2504.07952#S2.SS1), [Mem0 §2.1](https://arxiv.org/html/2504.19413#S2.SS1) and [AMEM §3](https://arxiv.org/html/2502.12110#S3). Hindsight runs its usual extraction on each retained record, with its LLM and embedding calls routed through the same backbone and embedder as everything else, and its recalled facts are packed under the same cap.

**Three rules that make the comparison fair.**

- *Same feedback for everyone.* After scoring, every memory system receives the question, the model's visible answer and the correctness bit, including failures. None of them ever sees the reference answer, so a lesson a curator writes is its own inference, not a verified solution. This makes DC a feedback-assisted variant of the original, which had no feedback at all.
- *Same budget.* The memory block is capped at 4,096 tokens, counted on the rendered text. Retrieval systems pack whole records until the cap; DC and ACE trim their sheet at bullet boundaries. The question, the system prompt and the answer are outside the cap and identical across systems.
- *Same model throughout.* The backbone that answers also runs every memory-side LLM call. The embedding model is shared and pinned. Memory resets for every stream, order, system and backbone.

**Why ExpRAG and dc-frozen are there.** ExpRAG is the control for *consolidation*: it stores and retrieves experience without transforming it, under the same cap. If a system that distils lessons cannot beat it, distillation added nothing. dc-frozen is the control for *updating*: it is DC's own end-state sheet, built from the whole stream, shown from question one and never changed. If online DC cannot beat its own frozen sheet, updating during the stream added nothing. A matched cap does not make the comparison perfectly causal, since the systems still differ in representation, retrieval and the histories they generate, but these two controls answer the two questions §2.4 asked.

### 5.2 What happened

Accuracy is exact match; no judge is involved in Part B.

**The short version: nothing learned.** On no stream, for no system, on either backbone, does accuracy rise over the stream more than it does for the no-memory baseline. Where memory helps, the help is there from the first questions and flat after that.

**Final accuracy, open model (three orders averaged on MMLU-Pro).**

| | none | exprag | dc | dc-frozen | mem0 | amem | ace | hindsight |
|---|---|---|---|---|---|---|---|---|
| Economics | 0.885 | 0.887 | 0.884 | 0.882 | 0.889 | 0.887 | 0.887 | **0.892** |
| Engineering | 0.666 | 0.684 | 0.673 | 0.680 | 0.674 | **0.685** | 0.676 | 0.680 |
| Philosophy | 0.756 | **0.787** | 0.752 | 0.758 | 0.768 | 0.777 | 0.766 | 0.774 |
| GPQA-Diamond | 0.692 | 0.682 | 0.702 | 0.682 | 0.682 | 0.707 | 0.662 | **0.727** |
| AIME-24 / 25 | 0.43 / 0.50 | 0.50 / 0.50 | 0.67 / 0.60 | **0.87** / 0.70 | 0.67 / 0.60 | 0.70 / 0.53 | 0.70 / 0.60 | 0.67 / 0.70 |

**Final accuracy, Sonnet (one order).**

| | none | exprag | dc | dc-frozen | mem0 | amem | ace | hindsight |
|---|---|---|---|---|---|---|---|---|
| Philosophy | 0.826 | 0.838 | **0.846** | **0.846** | 0.834 | 0.842 | 0.838 | 0.838 |
| GPQA-Diamond | 0.783 | 0.783 | 0.823 | 0.813 | 0.788 | 0.773 | 0.768 | **0.838** |
| AIME-24 / 25 | 0.70 / 0.67 | **0.93** / **0.83** | 0.83 / 0.77 | **0.93** / **0.83** | 0.87 / 0.70 | 0.87 / 0.80 | 0.87 / 0.77 | 0.83 / 0.80 |

![Figure 4. Cumulative accuracy over the stream, open model, orders averaged. Dashed black is no memory.](/images/agentmemory/f4_curves_qwen27b.png)

![Figure 5. Final accuracy per stream, Sonnet. Dashed line is no memory.](/images/agentmemory/f5_static_vs_learned_sonnet5.png)

Three things to take from the tables.

**1. The curves are flat.** My learning signal is simple: accuracy on the second half of the stream minus the first half. For every memory system it stays inside the range the *no-memory* run shows across question orders (about ±0.05), and with roughly 250 questions per half the noise floor is about 0.02. Nothing clears it. Expectation 5 fails: DC and ACE, the systems that write distilled lessons, curve no more than the ones that just retrieve.

**2. The gain is advice, not learning.** DC's final sheet, shown from question one and never updated, does as well as DC run online, on every stream and both backbones (within 0.02). So the curator's work during the stream adds nothing beyond its end state. Stranger still, on Sonnet GPQA a sheet DC wrote on *philosophy* questions helps graduate-level science questions exactly as much as the sheet written on those science questions: 160 versus 161 correct out of 198, both about six above no memory. Whatever the sheet contains, it is general problem-solving advice, not knowledge of the stream. This matches what [Zhao et al.](https://arxiv.org/abs/2601.22436) found with a different control and a different backbone. The effect is small, five to eight questions, but it is consistent across all three versions of the sheet.

**3. The simplest system is the best one.** ExpRAG, which retrieves raw past question-answer-verdict records with no LLM in the loop, is best or tied on all three MMLU-Pro streams on the open model and on both AIME streams on Sonnet. DC spends about 3.5K memory-side tokens per question rewriting its sheet, and on Philosophy (open model) lands at or below no memory. On the hardest reasoning stream the picture is mixed and the gaps are within noise.

**What this means.** On these streams, "self-evolving" memory is retrieval of past experience plus a static piece of advice, and neither one learns. The curves do not bend; the distilled sheet is interchangeable with one from another subject; the undistilled store is at least as good. Expectation 2 does not hold either: Sonnet gains as much from memory as the open model, and on AIME more, though the open model thinks and Sonnet does not, so the two differ in more than strength. §5.3 asks whether a task with real feedback, a build and hidden tests, changes any of this.

### 5.3 The coding stream

The question streams give memory one bit per question. A coding task gives more: a build that fails, tests that fail, a trace of tool calls. I ran the same protocol on a private set of single-module coding tasks graded by hidden tests. The agent was Claude Code, started with its own persistent memory switched off, with the memory block passed in as an extra system prompt. The open model was the backbone. Three conditions ran on the same task order: no memory, DC and ACE. After each task the memory received a summary of the trace and the grader's verdict, including which stage failed. The task set is not public, so I report the shape of the result without its numbers. The harness is in the repository and runs on any task set that provides a starting workspace and a grader.

**Whether tasks get solved: no detectable effect.** The no-memory agent already solves most of the tasks. The few that DC or ACE newly solve are balanced by ones they lose, and a paired test is nowhere near significance for either.

**How they get solved: a clear effect, and the two memories differ.** On the tasks every condition solves, both memories cut the number of turns, tool calls and input tokens. DC's saving is the same size in the first and second half of the stream. ACE's saving grows from the first half to the second, and its wall-clock time goes from slower than no memory early to faster late. This is the only place in the study where something improves with position in the stream.

That matches the contrast drawn in §2.1. A whole-sheet rewrite converges on advice that is useful from the first task and then stops changing; incremental edits with counters may be accumulating procedure specific to these tasks. I have not inspected the playbooks to confirm that, so it is a hypothesis. The caveats are real: one task order, one backbone, a comparison restricted to tasks every condition solved, and a memory that costs more memory-side tokens than DC. It is a lead that needs replicating on a public agentic benchmark.

## 6. Checking the judge, and how memory fails

### 6.1 Which scorer to trust

§3 left a question open: the official F1 says the open model answers LoCoMo better than Sonnet, and the judge says the opposite. To settle it, I drew 100 LoCoMo answers, a quarter from each combination of backbone and judge verdict, and labelled them correct or wrong myself, seeing only the question, the gold answer and the model's answer. I labelled 98.

| | judge | official F1 | judge too lenient / too strict | F1 too lenient / too strict |
|---|---|---|---|---|
| κ with my labels, all 98 | **0.755** | 0.601 | 9 / 3 | 7 / 12 |
| Sonnet's answers (49) | 0.795 | 0.528 | 4 / 1 | 3 / 8 |
| open model's answers (49) | 0.715 | 0.670 | 5 / 2 | 4 / 4 |

![Figure 6. Agreement with blind human labels, by scorer and by whose answers were scored.](/images/agentmemory/f6_judge_kappa.png)

**The judge is the better scorer.** Its agreement with my labels is substantial; the F1's is moderate. The gap is widest on temporal questions, where §3 found the two scorers furthest apart: κ 0.58 for the judge against 0.30 for the F1.

**The two scorers fail in opposite directions.** The judge's errors are lenient: it accepts near misses. The F1's errors are strict, and eight of its twelve strict errors are on Sonnet's answers, which are correct but longer than the gold phrase.

**The judge is not favouring its own family.** Of the answers it called correct, I marked wrong 4 of 24 from Sonnet and 5 of 25 from the open model. So the F1's lead for the open model came from terser answers matching the gold string, and the judge's lead for Sonnet survives the check.

Re-running the judge on the same 100 answers changed one verdict. Its leniency is consistent, not random, so re-judging would not remove it; a stricter rubric would.

The limits: one annotator, 98 items, LoCoMo only. BEAM's graded judge and LongMemEval's templates were not checked against humans.

### 6.2 How memory fails

- **Never written down.** Systems that extract facts keep what the user said and drop what the assistant said: Mem0 0.66 and Hindsight 0.18 on those LongMemEval questions, against 1.00 for full context and AMEM.
- **Written down, not retrieved.** BM25 says "not in the history" on a fifth of the answerable LongMemEval questions. The model is being honest about a block that missed the evidence.
- **Retrieved, but stale.** Mem0's extracted facts lose which fact came later: 0.57 on temporal questions and 0.73 on knowledge updates, against 0.81 and 0.85 for full context.
- **Cut off.** At 1M tokens full context keeps only the most recent window, and BEAM's knowledge-update score falls to zero because the original fact is no longer in the prompt.
- **Present but generic.** DC's cheatsheet from a different subject helps as much as the one from the same stream (§5.2). Whatever the sheet holds, it is not knowledge of the stream.
- **Out of budget.** On the open model a quarter of the hard reasoning questions exhaust the thinking budget and score zero whatever is in memory.

## 7. Limits and open problems

### 7.1 What this study can't tell you

- Every Part A number is a single run, and the Sonnet slices are small (20 to 100 LongMemEval questions, 3 to 5 BEAM conversations). In a few places slices of different sizes are compared without pairing.
- In Part B the open model thinks with a 16K budget and Sonnet does not, so backbone comparisons mix model strength with thinking. A quarter of the hard questions on the open model are lost to that budget.
- The 4,096-token memory cap is a choice. Hindsight, AMEM and BM25 fill it; Mem0 uses about a tenth of it.
- The memory systems run as shipped behind my adapters, with the row's own backbone writing the memory. Mem0 runs with a plain vector store, which turns its hybrid scoring off.
- dc-frozen uses the final sheet of the same stream, so it is an upper bound on what a static sheet can contribute. The other-subject control was run on one cell, and the effect it isolates is about one standard error.
- AIME has 30 questions per year, so one question moves the score by 0.033.
- BEAM's event-ordering ability is scored by LLM-matching answer lines to reference events; every system lands near the score of an answer that matches nothing, and I did not audit the matcher, so I draw nothing from it.
- Learning curves split by how similar each question is to earlier ones were not run; the frozen-sheet and other-subject controls stand in for them.
- The judge audit has one annotator and covers LoCoMo only. The coding stream is private and reported without numbers.

### 7.2 Open questions

**What would count as evidence that an agent learned during a stream?** In this study, the gap between two orderings of the same questions was larger than any gain a memory system showed over the stream. So a curve that rises is not enough: the same curve rises for no memory. Evidence would have to survive several orderings, show up on questions the memory never saw, and disappear when the memory is emptied or corrupted. I do not know of a memory paper that reports all three.

**What would a better judge look like?** The LLM judge beat the official scorer against human labels, but it won by being lenient: it accepts near-misses, and a judge that accepts near-misses will also accept a confident wrong answer on the right topic. I only tested it on real answers, not on answers written to fool it. A better judge would be checked both ways, on real answers and on deliberately wrong ones, would give partial credit on a scale that was itself validated, and would report how often it changes its mind on a re-run. None of the three benchmarks here ships a judge with those checks, so every memory result still carries a hidden choice of scorer.

**Where does memory start to beat reading everything?** This study has two points: at 100K tokens full context wins, at 1M it loses. The crossover is somewhere in between, and it almost certainly moves with the model, the task and the scorer, so the answer is probably a rule rather than a number: read everything while it fits, fall back to memory when it does not. What that rule should look like for conversations, with today's models and real histories between those two lengths, is still an open question.

## References

- LoCoMo — Maharana et al., *Evaluating Very Long-Term Conversational Memory of LLM Agents*, ACL 2024. [arXiv:2402.17753](https://arxiv.org/abs/2402.17753)
- LongMemEval — Wu et al., *LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory*, ICLR 2025. [arXiv:2410.10813](https://arxiv.org/abs/2410.10813)
- BEAM — Tavakoli et al., *Beyond a Million Tokens: Benchmarking and Enhancing Long-Term Memory in LLMs*, ICLR 2026. [arXiv:2510.27246](https://arxiv.org/abs/2510.27246)
- Evo-Memory — *Benchmarking LLM Agent Test-time Learning with Self-Evolving Memory* (the stream protocol and ExpRAG). [arXiv:2511.20857](https://arxiv.org/abs/2511.20857)
- EvoMemBench — *Benchmarking Agent Memory from a Self-Evolving Perspective*. [arXiv:2605.18421](https://arxiv.org/abs/2605.18421)
- Task streams: MMLU-Pro, Wang et al. 2024, [arXiv:2406.01574](https://arxiv.org/abs/2406.01574) · GPQA, Rein et al. 2023, [arXiv:2311.12022](https://arxiv.org/abs/2311.12022) · AIME 2024 and 2025 (American Invitational Mathematics Examination)

- Mem0 — Chhikara et al., *Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory*. [arXiv:2504.19413](https://arxiv.org/abs/2504.19413)
- AMEM — Xu et al., *A-MEM: Agentic Memory for LLM Agents*. [arXiv:2502.12110](https://arxiv.org/abs/2502.12110)
- Dynamic Cheatsheet — Suzgun et al., *Dynamic Cheatsheet: Test-Time Learning with Adaptive Memory*. [arXiv:2504.07952](https://arxiv.org/abs/2504.07952)
- ACE — Zhang et al., *Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models*. [arXiv:2510.04618](https://arxiv.org/abs/2510.04618)

- ReasoningBank — Ouyang et al., *ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory*. [arXiv:2509.25140](https://arxiv.org/abs/2509.25140)
- Memory-R1, [arXiv:2508.19828](https://arxiv.org/abs/2508.19828)
- MemAct, [arXiv:2510.12635](https://arxiv.org/abs/2510.12635)
- Mem-α, [arXiv:2509.25911](https://arxiv.org/abs/2509.25911)
- MemSearcher, [arXiv:2511.02805](https://arxiv.org/abs/2511.02805)
- MemEvolve, [arXiv:2512.18746](https://arxiv.org/abs/2512.18746)
- EvoAgentBench, [arXiv:2607.05202](https://arxiv.org/abs/2607.05202)
- Memento, [arXiv:2508.16153](https://arxiv.org/abs/2508.16153)
- GEPA, [arXiv:2507.19457](https://arxiv.org/abs/2507.19457)
- Zhao et al., *Large Language Model Agents Are Not Always Faithful Self-Evolvers*, ICML 2026. [arXiv:2601.22436](https://arxiv.org/abs/2601.22436)
