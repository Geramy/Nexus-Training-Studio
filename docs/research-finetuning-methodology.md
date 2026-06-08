{
  "summary": "Deep research harness — fan-out web searches, fetch sources, adversarially verify claims, synthesize a cited report.",
  "agentCount": 108,
  "logs": [
    "Q: Best practices and concrete hyperparameters for LoRA/QLoRA fine-tuning a tool-ca…",
    "Decomposed into 6 angles: broad/primary — LoRA hyperparameter defaults, module/layer targeting + MoE/Mamba, failure modes — forgetting/overfitting/quantization, evaluation & tool-call metrics, MLX / mlx-lm + MoE/hybrid specifics, model card / vendor guidance",
    "evaluation & tool-call metrics: 6 results",
    "failure modes — forgetting/overfitting/quantization: 6 results",
    "broad/primary — LoRA hyperparameter defaults: 6 results",
    "broad/primary — LoRA hyperparameter defaults: 3 novel (3 filtered)",
    "MLX / mlx-lm + MoE/hybrid specifics: 6 results",
    "MLX / mlx-lm + MoE/hybrid specifics: 4 novel (2 filtered)",
    "module/layer targeting + MoE/Mamba: 6 results",
    "module/layer targeting + MoE/Mamba: 3 novel (3 filtered)",
    "model card / vendor guidance: 6 results",
    "model card / vendor guidance: 3 novel (3 filtered)",
    "Fetched 25 sources → 116 claims → verifying top 25",
    "\"For MoE models, the authors trained a separate LoR…\": 3-0 ✓",
    "\"LoRA should be applied to ALL weight matrices, esp…\": 3-0 ✓",
    "\"LoRA matches full fine-tuning when applied to all …\": 1-2 ✗",
    "\"The optimal learning rate for LoRA is consistently…\": 2-1 ✓",
    "\"Applying LoRA to only query/value (or even just at…\": 3-0 ✓",
    "\"The NVIDIA Megatron Bridge LoRA API defaults targe…\": 3-0 ✓",
    "\"LoRA rank (projection dimension r) and other LoRA …\": 0-3 ✗",
    "\"NVIDIA's default LoRA rank (dim) is 32 and default…\": 0-3 ✗",
    "\"4-bit QLoRA with NF4 + double quantization fully m…\": 0-3 ✗",
    "\"LoRA substantially underperforms full fine-tuning …\": 3-0 ✓",
    "\"Nemotron 3 Nano is a Mixture-of-Experts hybrid Mam…\": 3-0 ✓",
    "\"LoRA mitigates catastrophic forgetting more effect…\": 3-0 ✓",
    "\"Parameter-efficient fine-tuning methods such as Lo…\": 2-1 ✓",
    "\"Catastrophic forgetting during LLM fine-tuning inc…\": 3-0 ✓",
    "\"There is a strong inverse linear relationship betw…\": 0-3 ✗",
    "\"A small, fixed set of just 1,000 general pretraini…\": 0-3 ✗",
    "\"Simply mixing in general replay samples during fin…\": 3-0 ✓",
    "\"BFCL evaluates function/tool calls using an Abstra…\": 3-0 ✓",
    "\"BFCL uses an AST (Abstract Syntax Tree) metric to …\": 3-0 ✓",
    "\"In mlx-lm LoRA, the number of layers adapted (--nu…\": 3-0 ✓",
    "\"mlx-lm automatically uses QLoRA (training against …\": 3-0 ✓",
    "\"Prolonged training and large learning rates cause …\": 3-0 ✓",
    "\"Unitxt's ToolCallingMetric uses exact_match as its…\": 3-0 ✓",
    "\"To avoid over-memorization, the authors recommend …\": 1-2 ✗",
    "\"For model/checkpoint selection, the authors recomm…\": 3-0 ✓",
    "Verify done: 25 claims → 18 confirmed, 7 killed"
  ],
  "result": {
    "question": "Best practices and concrete hyperparameters for LoRA/QLoRA fine-tuning a tool-calling assistant on Apple-Silicon MLX, specifically for a 30B-parameter HYBRID Mamba/Transformer Mixture-of-Experts model (NVIDIA Nemotron-3-Nano-30B-A3B, ~3B active params, attention modules named mixer.q_proj/k_proj/v_proj/o_proj, with shared_experts in the MoE). I need actionable, cited guidance I can put directly into an mlx_lm LoRA config and training plan:\n\n(1) HYPERPARAMETERS: recommended LoRA rank (r) and alpha (scaling) for instruction/tool-use fine-tunes, the alpha:r ratio convention, LoRA dropout, learning rate ranges for LoRA vs QLoRA (and why LoRA LR is higher than full-FT), warmup, LR schedule (cosine vs linear), batch size + gradient accumulation, and how many iterations/epochs before overfitting on a few hundred to a few thousand examples. Give numeric defaults.\n\n(2) WHICH MODULES/LAYERS TO TARGET: should LoRA target only attention projections (q/k/v/o) or also MLP/expert projections? For a Mixture-of-Experts model, the implications of applying LoRA to expert vs shared_expert vs router/gate layers; for a HYBRID Mamba+attention model, whether to adapt the Mamba mixer (in_proj/out_proj) too; how many layers (num_layers) to adapt vs all of them; trade-offs.\n\n(3) AVOIDING THE FAILURE MODES: catastrophic forgetting and how to prevent it (replay/general-data mixing %, lower LR, fewer epochs, lower rank); overfitting signs on small datasets and mitigations; \"model got dumber / repeats / lost general ability after LoRA\" — root causes and fixes; how merging/fusing a LoRA adapter and then QUANTIZING (to GGUF Q6/Q4) degrades quality and how to minimize it (de-quantize before fuse, imatrix, QLoRA vs LoRA serving).\n\n(4) EVALUATION & STOPPING: how to set validation split, read train vs val loss to pick the checkpoint, detect overfitting early, and quantify tool-call correctness (exact-match of function name + JSON args, schema-valid arguments, BFCL-style metrics). When to stop.\n\n(5) MoE/MLX SPECIFICS: any known issues fine-tuning MoE or hybrid-Mamba models with LoRA (expert imbalance, router drift), MLX/mlx-lm gotchas, QLoRA on Apple Silicon memory behavior, and whether to train at 8-bit base (quantized) vs bf16 and the quality difference.\n\nPrefer sources from 2024-2026 (Hugging Face PEFT docs, Unsloth, mlx-lm docs/issues, NVIDIA Nemotron model card, QLoRA paper, LoRA learning-rate and rank ablation studies). Give specific numbers and a recommended starting config.",
    "summary": "For a LoRA/QLoRA fine-tune of a 30B hybrid Mamba/Transformer MoE (Nemotron-3-Nano-30B-A3B) on a few hundred to a few thousand tool-use examples in mlx-lm, the strongest, most consistent finding across primary research (Thinking Machines \"LoRA Without Regret\", QLoRA paper, \"LoRA Learns Less and Forgets Less\") is: apply LoRA to ALL linear layers — especially MLP and the MoE expert projections — not attention-only, because attention-only LoRA fails to match full fine-tuning (attention-only r=256 even underperforms MLP-only r=128). Set the LoRA learning rate ~10x higher than you would for full fine-tuning (~15x for very short <100-step runs) and treat it as roughly rank-independent. LoRA forgets less general capability than full fine-tuning and beats weight-decay/dropout as a forgetting mitigation, but it does NOT escape catastrophic forgetting — forgetting rises monotonically (shifted power law) with both trainable-parameter count and update steps, so prefer fewer epochs, modest rank/LR, and mix in general \"replay\" data (which gave a 12-point MMLU recovery in a continual-FT study). Watch for \"over-memorization\" (test accuracy holds while test perplexity rises, hurting robustness/OOD/diversity) driven by long training and high LR; select checkpoints by highest validation accuracy within a bounded validation-perplexity range, and quantify tool-call quality with AST-based (BFCL) and exact-match (Unitxt) metrics on function name + arguments. mlx-lm specifics: --num-layers defaults to 16 (a subset, not all layers) and QLoRA is auto-enabled when --model points to a quantized checkpoint.",
    "findings": [
      {
        "claim": "Target ALL linear layers — especially MLP and MoE expert projections — not attention-only. Attention-only LoRA fails to match full fine-tuning; attention-only r=256 underperforms MLP-only r=128 at equal parameter count. For the MoE, train a separate LoRA per expert with each expert's rank = total_rank / number_of_active_experts. NVIDIA's own Megatron Bridge default targets attention+MLP (linear_qkv, linear_proj, linear_fc1, linear_fc2).",
        "confidence": "high",
        "vote": "3-0 (claims 1,2,3,4)",
        "sources": [
          "https://thinkingmachines.ai/blog/lora/",
          "https://arxiv.org/pdf/2305.14314",
          "https://arxiv.org/pdf/2405.09673",
          "https://docs.nvidia.com/nemo/megatron-bridge/latest/apidocs/bridge/bridge.peft.lora.html",
          "https://huggingface.co/docs/trl/en/lora_without_regret"
        ],
        "evidence": "Thinking Machines: 'Attention-only LoRA significantly underperforms MLP-only LoRA... attention-only with rank 256 underperforms MLP-only with rank 128, despite ~same number of parameters'; verified to hold on a sparse MoE (Qwen3-30B-A3B-Base). For MoE: 'we trained a separate LoRA on each expert, with the rank of each equal to the total rank divided by the number of active experts'. QLoRA paper: 'LoRA on all linear transformer block layers are required to match full finetuning performance.' Megatron Bridge default = ['linear_qkv','linear_proj','linear_fc1','linear_fc2'] (attention + MLP). Caveat: Nemotron has 128 routable experts / 6 activated + shared expert(s) and a hybrid Mamba layout, so the per-expert rank divisor (use active=6) and module names (mixer.q/k/v/o_proj, shared_experts, Mamba in_proj/out_proj) must be mapped to its specific architecture; the literature gives no Mamba-mixer-specific LoRA guidance."
      },
      {
        "claim": "Set the LoRA learning rate ~10x the optimal full-fine-tuning LR (~15x for short <100-step runs); the optimal LoRA LR is largely rank-independent.",
        "confidence": "medium",
        "vote": "2-1 (claim 0)",
        "sources": [
          "https://thinkingmachines.ai/blog/lora/",
          "https://arxiv.org/abs/2602.06204"
        ],
        "evidence": "Thinking Machines: 'the optimal LR for LoRA is consistently 10x the one used for FullFT... ~15x for short runs... optimal LR is also independent of rank.' Caveat: conditioned on the standard 1/r LoRA parametrization with no theoretical explanation; a 2026 paper (arXiv 2602.06204) shows the LoRA-to-FFT LR relationship is configuration-dependent (init + alpha) and under Init[B], alpha=1 the multiplier is ~1x, not 10x. Treat 10x as a starting heuristic to sweep, not a law. The split vote reflects this configuration-dependence."
      },
      {
        "claim": "LoRA forgets LESS general capability than full fine-tuning (and beats weight decay/dropout as a forgetting mitigation, preserving generation diversity), making LoRA itself a forgetting-mitigation lever — but it still underperforms full FT on hard in-domain learning (coding/math) at standard low ranks.",
        "confidence": "high",
        "vote": "3-0 (claims 6,7)",
        "sources": [
          "https://arxiv.org/pdf/2405.09673",
          "https://arxiv.org/abs/2410.21228"
        ],
        "evidence": "Biderman et al. (TMLR 2024): 'in the standard low-rank settings, LoRA substantially underperforms full finetuning... LoRA better maintains the base model's performance on tasks outside the target domain'; 'LoRA mitigates forgetting more than common regularization techniques such as weight decay and dropout; it also helps maintain more diverse generations.' Strengthened by arXiv 2410.21228 (forgets-less is intrinsic). Caveat: the 'no extra regularization needed' implication is slightly stronger than the evidence; OPLoRA (2510.13003) and 2512.17720 show bad hyperparameters can still induce forgetting."
      },
      {
        "claim": "LoRA does NOT escape catastrophic forgetting — it still loses prior abilities. Forgetting increases as a shifted power law in BOTH the number of trainable parameters and the number of update steps, so more trainable params and more iterations both monotonically increase forgetting.",
        "confidence": "high",
        "vote": "2-1 (claim 9) + 3-0 (claim 8)",
        "sources": [
          "https://arxiv.org/pdf/2401.05605",
          "https://arxiv.org/pdf/2405.09673"
        ],
        "evidence": "Scaling Laws for Forgetting (2401.05605): 'PEFT strategies, such as Low-Rank Adapters (LoRA), still suffer from catastrophic forgetting'; 'forgetting increases as a shifted power law in the number of parameters fine-tuned and the number of update steps' (positive exponents alpha_ft~0.04, beta_ft~0.12). Practical levers: lower rank, fewer steps/epochs, lower LR. Caveat: fit on Llama-2-7B-chat, not a 30B hybrid-Mamba MoE — directional, not exact transfer. Do NOT inflate to 'LoRA forgets as much as full FT' (it forgets less)."
      },
      {
        "claim": "Mix in general 'replay' data during fine-tuning to recover general capability. In a continual full-FT study, adding general replay samples raised MMLU from 38.32% to 50.53% (+12.21 points).",
        "confidence": "medium",
        "vote": "3-0 (claim 10)",
        "sources": [
          "https://arxiv.org/pdf/2508.04676"
        ],
        "evidence": "GeRe (arXiv 2508.04676), Table II: Baseline (no replay) MMLU 38.32 vs BaselineR (replay) 50.53, +12.21 pts. Caveat: measured on Llama-3.1-8B under FULL-parameter continual fine-tuning across 15 sequential tasks — NOT a single LoRA tool-use run on a 30B MoE; the 12-point magnitude should not be quoted as a guaranteed gain here. The refuted sub-claim ('1,000 fixed SlimPajama samples suffice', vote 0-3) means do not rely on a tiny fixed replay set. Direction (replay reduces forgetting) is robust; exact replay % is unspecified by surviving evidence."
      },
      {
        "claim": "Beware 'over-memorization': prolonged training and large learning rates cause test accuracy to hold while test perplexity rises, degrading robustness, OOD generalization, and generation diversity. Occurs in both full FT and LoRA. Fix: shorter training and lower LR.",
        "confidence": "high",
        "vote": "3-0 (claim 15)",
        "sources": [
          "https://arxiv.org/pdf/2508.04117"
        ],
        "evidence": "arXiv 2508.04117: 'prolonged training and large learning rates exacerbating the problem... models with over-memorization demonstrate comparable test accuracy... but suffer from reduced robustness, poor out-of-distribution generalization, and decreased generation diversity'; 'over-memorization occurs not only in LoRA but also across various finetuning methods.' Caveat: 2025 preprint, experiments on reasoning tasks (not tool-calling specifically)."
      },
      {
        "claim": "For checkpoint/early-stopping selection, combine validation accuracy and perplexity: pick the checkpoint with the highest validation accuracy within a bounded range of validation perplexity, rather than either metric alone (perplexity selects an earlier checkpoint, accuracy a later/over-memorized one).",
        "confidence": "high",
        "vote": "3-0 (claim 16)",
        "sources": [
          "https://arxiv.org/pdf/2508.04117"
        ],
        "evidence": "arXiv 2508.04117, Sec 7.1: 'choosing models with the highest validation accuracy within a certain range of validation perplexity.' Checkpoint averaging offered as an alternative. Note: the related 'limit to 1-4 epochs then use last checkpoint' claim was REFUTED (vote 1-2) — do not blindly use the last checkpoint; select by the val-accuracy/perplexity combination."
      },
      {
        "claim": "Quantify tool-call correctness with AST-based and exact-match metrics. BFCL uses an Abstract Syntax Tree (AST) matching method (checking function name + arguments / parameter types against ground truth and the formal signature) that scales to thousands of functions, beyond pure string match. Unitxt's ToolCallingMetric exposes a strict exact_match main score (full tool call = name + arguments) plus sub-metrics (tool_name_accuracy, argument name/value precision/recall, argument_schema_validation).",
        "confidence": "high",
        "vote": "3-0 (claims 11,12,17)",
        "sources": [
          "https://gorilla.cs.berkeley.edu/leaderboard.html",
          "https://proceedings.mlr.press/v267/patil25a.html",
          "https://www.unitxt.ai/en/main/docs/tool_calling.html"
        ],
        "evidence": "BFCL (ICML 2025): 'a novel Abstract Syntax Tree (AST) evaluation method that can easily scale to thousands of functions'; AST checks function identification, obligatory parameters, and parameter types/values as a proxy for execution. Unitxt: main_score = exact_match ('Measures if the tool call exactly matches a reference') with companion schema-validation and argument precision/recall metrics. Practical: track exact-match of (function name + JSON args), schema-validity of args, and AST/BFCL-style accuracy."
      },
      {
        "claim": "mlx-lm specifics: --num-layers defaults to 16 (LoRA applied to only a subset of layers, not all of them by default; lower to 8/4 to save memory at a quality cost), and QLoRA is auto-enabled — no separate flag — whenever --model points to a quantized checkpoint (otherwise regular LoRA).",
        "confidence": "high",
        "vote": "3-0 (claims 13,14)",
        "sources": [
          "https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md"
        ],
        "evidence": "LORA.md: 'Reduce the number of layers to fine-tune with --num-layers. The default is 16, so you can try 8 or 4... This reduces memory needed for back propagation' (may reduce quality with lots of data). 'If --model points to a quantized model, then the training will use QLoRA, otherwise it will use regular LoRA.' Implication for a 30B model: default num-layers=16 adapts only the top 16 transformer layers — for the all-layers coverage the research recommends you must raise num-layers, trading memory. Quantize the base beforehand via convert -q to get QLoRA."
      },
      {
        "claim": "Nemotron-3-Nano-30B-A3B is confirmed as a MoE hybrid Mamba-Transformer (~31.6B total / ~3.2B active per forward pass; 128 routable experts, 6 activated/token, plus shared expert(s); ~23 Mamba-2/MoE layers + ~6 GQA attention layers), validating that the plan must handle MoE expert/shared-expert/router modules AND Mamba mixer modules — not just attention.",
        "confidence": "high",
        "vote": "3-0 (claim 5)",
        "sources": [
          "https://arxiv.org/abs/2512.20848",
          "https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
        ],
        "evidence": "arXiv 2512.20848 title: 'Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning'; HF model card confirms 31.6B total / 3.2B active. Architecture: granular MoE (128 experts, 6 active) + shared expert(s) (sources vary 1 vs 2), 23 Mamba-2/MoE + 6 GQA attention layers. This grounds the per-expert LoRA rank divisor (use active experts = 6, not 8) and the existence of shared_experts and Mamba in_proj/out_proj to consider as targets."
      }
    ],
    "caveats": "Source-strength: the highest-leverage practical findings (apply LoRA to all/MLP/MoE layers; 10x LR; per-expert LoRA) lean heavily on ONE research-lab blog (Thinking Machines \"LoRA Without Regret\", not peer-reviewed), though corroborated by the peer-reviewed QLoRA paper and TMLR \"LoRA Learns Less and Forgets Less\". The 10x LR rule got a split 2-1 vote and is explicitly configuration-dependent (init/alpha) per a 2026 paper — a starting heuristic to sweep, not a law. Transfer gap: nearly all forgetting/replay/over-memorization numbers come from dense models (Llama-2/3.1-7-8B, Qwen3-30B MoE) under different regimes (continual multi-task FT, reasoning tasks) — none from a 30B hybrid-Mamba MoE on tool-calling, so magnitudes (12-pt MMLU recovery, power-law exponents) are directional, not guaranteed. Architecture-specific void: NO surviving source gives concrete guidance on adapting the Mamba mixer (in_proj/out_proj), router/gate layers, or shared_experts under LoRA, nor on the exact alpha:r ratio or LoRA dropout (the NVIDIA alpha=r=32 1:1-ratio claim was REFUTED 0-3, so do not treat 1:1 as an institutional default). The claims 'QLoRA fully matches 16-bit' and 'rank does not matter' were REFUTED (0-3), so do not assume an 8-bit/4-bit base is loss-free vs bf16, and do not assume rank is irrelevant. Time-sensitivity: mlx-lm is fast-moving (main-branch defaults like num-layers=16 can change) — verify against the installed version. GGUF post-fuse quantization (Q6/Q4), imatrix, and de-quantize-before-fuse questions from the original prompt were NOT covered by any surviving claim.",
    "openQuestions": [
      "What concrete alpha:r ratio, LoRA dropout, warmup, cosine-vs-linear schedule, and batch/grad-accum should be used for this MoE in mlx-lm? No surviving claim gives these numbers (the NVIDIA alpha=r=32 default was refuted), so they must be set/swept empirically.",
      "Should LoRA adapt the Mamba mixer (in_proj/out_proj), the router/gate, and the shared_experts — and at what rank — for a hybrid Mamba+MoE model? No source addresses Mamba-mixer or router/gate LoRA directly; per-expert rank = total/active is established but shared-expert and router handling is unknown.",
      "How much does fusing a LoRA adapter and then quantizing to GGUF Q6/Q4 degrade tool-calling quality, and do de-quantize-before-fuse / imatrix / serving QLoRA vs fused-LoRA mitigate it? Not covered by any surviving claim — needs direct empirical testing.",
      "What replay percentage (general-data mix) and exact iteration/epoch budget minimize forgetting for a single-domain LoRA tool-use run on a few hundred to a few thousand examples? Surviving evidence confirms replay and fewer steps help, but gives no validated percentage or epoch count for this regime (the 1-4-epoch and 1,000-fixed-sample recipes were refuted)."
    ],
    "refuted": [
      {
        "claim": "LoRA matches full fine-tuning when applied to all weight matrices and trainable parameters exceed the information in the dataset (small-to-medium instruction/reasoning datasets like Tulu3, OpenThoughts3); it underperforms only when the dataset exceeds LoRA capacity.",
        "vote": "1-2",
        "source": "https://thinkingmachines.ai/blog/lora/"
      },
      {
        "claim": "LoRA rank (projection dimension r) and other LoRA hyperparameters do NOT affect performance, contrary to the assumption that rank needs careful tuning; what matters is breadth of adapter coverage, not rank.",
        "vote": "0-3",
        "source": "https://arxiv.org/pdf/2305.14314"
      },
      {
        "claim": "4-bit QLoRA with NF4 + double quantization fully matches 16-bit full finetuning and 16-bit LoRA finetuning performance, meaning training on a quantized base does not degrade quality versus bf16 when adapters are added at all layers.",
        "vote": "0-3",
        "source": "https://arxiv.org/pdf/2305.14314"
      },
      {
        "claim": "NVIDIA's default LoRA rank (dim) is 32 and default alpha is 32, giving an alpha:r ratio of 1:1 as the institutional default for these models.",
        "vote": "0-3",
        "source": "https://docs.nvidia.com/nemo/megatron-bridge/latest/apidocs/bridge/bridge.peft.lora.html"
      },
      {
        "claim": "There is a strong inverse linear relationship between fine-tuning task performance and the amount of forgetting under LoRA, implying you cannot push task performance higher without proportionally increasing degradation of general ability.",
        "vote": "0-3",
        "source": "https://arxiv.org/pdf/2401.05605"
      },
      {
        "claim": "A small, fixed set of just 1,000 general pretraining samples (randomly selected from SlimPajama-627B) is sufficient to mitigate catastrophic forgetting during continual fine-tuning of LLMs.",
        "vote": "0-3",
        "source": "https://arxiv.org/pdf/2508.04676"
      },
      {
        "claim": "To avoid over-memorization, the authors recommend limiting fine-tuning to a small number of epochs, typically between 1 and 4, after which the last checkpoint can be used as the final model.",
        "vote": "1-2",
        "source": "https://arxiv.org/pdf/2508.04117"
      }
    ],
    "sources": [
      {
        "url": "https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms",
        "quality": "blog",
        "angle": "broad/primary — LoRA hyperparameter defaults",
        "claimCount": 5
      },
      {
        "url": "https://thinkingmachines.ai/blog/lora/",
        "quality": "primary",
        "angle": "broad/primary — LoRA hyperparameter defaults",
        "claimCount": 5
      },
      {
        "url": "https://arxiv.org/pdf/2305.14314",
        "quality": "primary",
        "angle": "broad/primary — LoRA hyperparameter defaults",
        "claimCount": 5
      },
      {
        "url": "https://docs.nvidia.com/nemo/megatron-bridge/latest/apidocs/bridge/bridge.peft.lora.html",
        "quality": "primary",
        "angle": "module/layer targeting + MoE/Mamba",
        "claimCount": 5
      },
      {
        "url": "https://www.ikangai.com/lora-without-regret-a-practitioners-guide-to-reliable-fine-tuning/",
        "quality": "secondary",
        "angle": "module/layer targeting + MoE/Mamba",
        "claimCount": 5
      },
      {
        "url": "https://arxiv.org/abs/2512.20848",
        "quality": "primary",
        "angle": "module/layer targeting + MoE/Mamba",
        "claimCount": 3
      },
      {
        "url": "https://kaitchup.substack.com/p/dont-merge-your-lora-adapter-into",
        "quality": "blog",
        "angle": "failure modes — forgetting/overfitting/quantization",
        "claimCount": 4
      },
      {
        "url": "https://salt.sunbird.ai/sunflower/quantization/",
        "quality": "secondary",
        "angle": "failure modes — forgetting/overfitting/quantization",
        "claimCount": 5
      },
      {
        "url": "https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide",
        "quality": "secondary",
        "angle": "failure modes — forgetting/overfitting/quantization",
        "claimCount": 5
      },
      {
        "url": "https://arxiv.org/pdf/2405.09673",
        "quality": "primary",
        "angle": "failure modes — forgetting/overfitting/quantization",
        "claimCount": 3
      },
      {
        "url": "https://arxiv.org/pdf/2401.05605",
        "quality": "primary",
        "angle": "failure modes — forgetting/overfitting/quantization",
        "claimCount": 5
      },
      {
        "url": "https://arxiv.org/pdf/2508.04676",
        "quality": "primary",
        "angle": "failure modes — forgetting/overfitting/quantization",
        "claimCount": 4
      },
      {
        "url": "https://gorilla.cs.berkeley.edu/leaderboard.html",
        "quality": "primary",
        "angle": "evaluation & tool-call metrics",
        "claimCount": 4
      },
      {
        "url": "https://proceedings.mlr.press/v267/patil25a.html",
        "quality": "primary",
        "angle": "evaluation & tool-call metrics",
        "claimCount": 4
      },
      {
        "url": "https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md",
        "quality": "primary",
        "angle": "evaluation & tool-call metrics",
        "claimCount": 5
      },
      {
        "url": "https://arxiv.org/pdf/2508.04117",
        "quality": "primary",
        "angle": "evaluation & tool-call metrics",
        "claimCount": 5
      },
      {
        "url": "https://futureagi.com/blog/evaluating-tool-calling-agents-2026/",
        "quality": "blog",
        "angle": "evaluation & tool-call metrics",
        "claimCount": 5
      },
      {
        "url": "https://www.unitxt.ai/en/main/docs/tool_calling.html",
        "quality": "primary",
        "angle": "evaluation & tool-call metrics",
        "claimCount": 5
      },
      {
        "url": "https://github.com/ml-explore/mlx-lm/issues/571",
        "quality": "forum",
        "angle": "MLX / mlx-lm + MoE/hybrid specifics",
        "claimCount": 4
      },
      {
        "url": "https://github.com/ARahim3/mlx-tune",
        "quality": "secondary",
        "angle": "MLX / mlx-lm + MoE/hybrid specifics",
        "claimCount": 5
      },
      {
        "url": "https://github.com/unslothai/unsloth/discussions/3810",
        "quality": "forum",
        "angle": "MLX / mlx-lm + MoE/hybrid specifics",
        "claimCount": 5
      },
      {
        "url": "https://arxiv.org/pdf/2512.20848",
        "quality": "primary",
        "angle": "MLX / mlx-lm + MoE/hybrid specifics",
        "claimCount": 5
      },
      {
        "url": "https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        "quality": "primary",
        "angle": "model card / vendor guidance",
        "claimCount": 5
      },
      {
        "url": "https://docs.nvidia.com/nemo/megatron-bridge/latest/models/llm/nemotron3.html",
        "quality": "primary",
        "angle": "model card / vendor guidance",
        "claimCount": 5
      },
      {
        "url": "https://arxiv.org/html/2512.20848v1",
        "quality": "primary",
        "angle": "model card / vendor guidance",
        "claimCount": 5
      }
    ],
    "stats": {
      "angles": 6,
      "sourcesFetched": 25,
      "claimsExtracted": 116,
      "claimsVerified": 25,
      "confirmed": 18,
      "killed": 7,
      "afterSynthesis": 10,
      "urlDupes": 4,
      "budgetDropped": 7,
      "agentCalls": 108
    }
  }
}