{
  "summary": "Deep research harness — fan-out web searches, fetch sources, adversarially verify claims, synthesize a cited report.",
  "agentCount": 112,
  "logs": [
    "Q: How to build high-quality fine-tuning data for a TOOL-CALLING (function-calling)…",
    "Decomposed into 6 angles: tool-calling fine-tune data format & loss masking (primary technical), LoRA/QLoRA dataset size, diversity & error-recovery examples (empirical numbers), mlx-lm LoRA JSONL chat format & long-context config (implementation), long-context training: sequence packing, truncation, decontamination, requirements discovery interview technique (PM/UX/consulting), user stories, INVEST, Gherkin acceptance criteria & story mapping",
    "requirements discovery interview technique (PM/UX/consulting): 6 results",
    "user stories, INVEST, Gherkin acceptance criteria & story mapping: 6 results",
    "LoRA/QLoRA dataset size, diversity & error-recovery examples (empirical numbers): 6 results",
    "LoRA/QLoRA dataset size, diversity & error-recovery examples (empirical numbers): 4 novel (2 filtered)",
    "long-context training: sequence packing, truncation, decontamination: 6 results",
    "long-context training: sequence packing, truncation, decontamination: 5 novel (1 filtered)",
    "tool-calling fine-tune data format & loss masking (primary technical): 6 results",
    "tool-calling fine-tune data format & loss masking (primary technical): 5 novel (1 filtered)",
    "mlx-lm LoRA JSONL chat format & long-context config (implementation): 6 results",
    "mlx-lm LoRA JSONL chat format & long-context config (implementation): 3 novel (3 filtered)",
    "Fetched 29 sources → 140 claims → verifying top 25",
    "\"Loss masking to train only on assistant messages i…\": 3-0 ✓",
    "\"For prompt-completion datasets, TRL computes loss …\": 3-0 ✓",
    "\"TRL's SFTTrainer fully supports fine-tuning tool-c…\": 3-0 ✓",
    "\"The minimum number of training examples accepted i…\": 3-0 ✓",
    "\"OpenAI's fine-tuning data uses the chat completion…\": 2-1 ✓",
    "\"Pre-collected (static) error-correction datasets d…\": 3-0 ✓",
    "\"A synthetic data generation method that optimizes …\": 3-0 ✓",
    "\"mlx_lm.lora natively supports a 'tools' JSONL data…\": 3-0 ✓",
    "\"Loss masking of the prompt (training only on the c…\": 3-0 ✓",
    "\"Smaller tool-calling LLMs frequently fall into rep…\": 3-0 ✓",
    "\"mlx_lm.lora supports four JSONL data formats: chat…\": 3-0 ✓",
    "\"Existing function-calling synthetic-data work has …\": 3-0 ✓",
    "\"For the chat, tools, and completions formats, mlx-…\": 3-0 ✓",
    "\"The official mlx-lm lora_config.yaml defaults max_…\": 3-0 ✓",
    "\"Packing dramatically reduces SFT training time on …\": 3-0 ✓",
    "\"Packing benefits grow with model size: the perform…\": 3-0 ✓",
    "\"The default LoRA hyperparameters in the official c…\": 2-1 ✓",
    "\"Sequence packing yields a 2x speedup for phase 2 B…\": 3-0 ✓",
    "\"Packing/preparation works by randomly sampling and…\": 2-1 ✓",
    "\"Packing multiple sequences into one example requir…\": 3-0 ✓",
    "\"For long-context continued training, the best aver…\": 3-0 ✓",
    "\"Open-ended questions are best for discovery interv…\": 3-0 ✓",
    "\"Discovery interviews should use a funnel technique…\": 3-0 ✓",
    "\"Effective user interviews use open-ended questions…\": 3-0 ✓",
    "\"Cross-document attention should be disabled (docum…\": 2-1 ✓",
    "Verify done: 25 claims → 25 confirmed, 0 killed"
  ],
  "result": {
    "question": "How to build high-quality fine-tuning data for a TOOL-CALLING (function-calling) LLM agent, AND how product teams run requirements/discovery interviews and write user stories — so I can synthesize a large, realistic training corpus. Cover with concrete, cited, actionable guidance:\n\n(A) TRAINING DATA FOR TOOL-CALLING AGENTS: the OpenAI/messages format for multi-turn function-calling traces (system/user/assistant-with-tool_calls/tool-result); LOSS MASKING / completion-only training (train only on assistant + tool-call tokens, mask user/system/tool-result) and whether it matters for LoRA; how MANY examples are actually needed for a LoRA/QLoRA fine-tune to reliably learn tool-call FORMAT and behavior (hundreds vs thousands), and the diversity/quality tradeoffs; whether to include NEGATIVE / error-recovery examples (failed tool call → retry); multi-tool and multi-step sequences; how to handle LONG CONTEXT (32k–64k) examples — sequence packing, truncation, sample length distribution, and memory/throughput implications on Apple-Silicon MLX; data dedup and decontamination; balancing multiple skills/agents in ONE model so they don't interfere.\n\n(B) MLX-LM SPECIFICS: the exact JSONL chat format mlx_lm.lora expects (does it accept {\"messages\":[...]} with tool_calls? does it support a tools field / chat-template tool rendering?), how mlx-lm applies the chat template and whether it masks the prompt, and the config knobs (max_seq_length, lora layers, iters, batch size) relevant to 32–64k context on a 30B MoE.\n\n(C) DISCOVERY / REQUIREMENTS INTERVIEW TECHNIQUE (so synthetic interviewer turns are realistic): how PMs, consultants, and UX researchers actually elicit requirements — funnel from broad to specific, open vs closed questions, the \"5 whys\", probing for edge/error cases and unstated assumptions, MoSCoW prioritization, and one-question-at-a-time pacing.\n\n(D) USER STORIES & STORY MAPPING: INVEST criteria, the \"As a <role>, I want <goal>, so that <benefit>\" template, acceptance criteria (Gherkin/Given-When-Then), epics→stories→sub-stories hierarchy, user story MAPPING (building a tree/backbone of a flow), and how to decompose a story into well-scoped engineering TASKS with verification.\n\nGive concrete examples and source-backed numbers (e.g. recommended example counts, packing strategies) I can apply directly.",
    "summary": "To build a realistic tool-calling training corpus, use the OpenAI/messages JSONL format where each line carries a `messages` array (system/user/assistant-with-tool_calls/tool-result) plus a per-example `tools` array of JSON-schema function definitions — this exact shape is natively supported by mlx_lm.lora's \"tools\" format and by TRL's SFTTrainer. Train completion-only (mask user/system/tool-result, learn only assistant + tool-call tokens): in mlx-lm via `--mask-prompt` (chat/completion only), in TRL via `assistant_only_loss=True`/`completion_only_loss=True`; masking matters even for LoRA. Quantities are modest — OpenAI accepts 10, recommends ~50, and sees gains at 50-100 — but diversity is the real lever: recent research (arXiv 2601.17829) shows optimizing linguistic diversity of requests and argument-value coverage yields +7.4% on BFCL, and error-recovery examples are essential because small models otherwise loop on failed calls (arXiv 2601.15625). For 32-64k context on Apple Silicon, the mlx-lm `max_seq_length` default of 2048 must be overridden, sequence packing with document-boundary attention masking gives large speedups (2x BERT phase-2; 60-85% time cuts on big models) while preserving quality, and a 60% long / 40% short data mix is the validated default (arXiv 2410.02660). For realistic synthetic discovery/interview and user-story turns, use NN/g-backed funnel technique (broad open-ended → specific closed) and experience-eliciting prompts.",
    "findings": [
      {
        "claim": "The canonical tool-calling fine-tuning format is chat-completions JSONL: each line is a JSON object with a `messages` array (system/user/assistant-with-tool_calls/tool-result) PLUS a per-example `tools` array of JSON-schema function definitions; assistant turns carry tool_calls (id/type/function with name and JSON-string arguments). This exact format is supported by OpenAI, TRL's SFTTrainer, and mlx_lm.lora's 'tools' format.",
        "confidence": "high",
        "vote": "consensus (claims 0:2-1, 2:3-0, 9:3-0, 10:3-0)",
        "evidence": "OpenAI docs require the complete `tools` array per example for function-calling fine-tunes. TRL SFTTrainer 'fully supports' tool calling with conversation messages (tool_calls + tool-role responses) plus a `tools` column of JSON schemas. mlx_lm.lora natively accepts {\"messages\":[...],\"tools\":[...]} with the verbatim get_current_weather example, and supports four JSONL formats: chat, tools, completions, text. Note: arguments may be JSON strings vs dicts depending on model (OpenAI/Mistral/HF conventions).",
        "sources": [
          "https://platform.openai.com/docs/guides/supervised-fine-tuning",
          "https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/fine-tuning-functions",
          "https://huggingface.co/docs/trl/sft_trainer",
          "https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md"
        ]
      },
      {
        "claim": "Completion-only / loss masking (train only on assistant + tool-call tokens, ignore user/system/tool-result) is the recommended approach and is supported across frameworks. In mlx-lm use `--mask-prompt` (chat and completion datasets only); in TRL use assistant_only_loss=True (requires {% generation %}/{% endgeneration %} chat-template keywords, auto-patched for Qwen3) or completion_only_loss=True (the default for prompt-completion datasets). The two can be combined.",
        "confidence": "high",
        "vote": "consensus (claims 3:3-0, 4:3-0, 11:3-0)",
        "evidence": "TRL: assistant_only_loss=True computes loss only on assistant responses, ignoring user/system; completion_only_loss is True by default for prompt-completion datasets; a TIP confirms combining both on a conversational prompt-completion dataset. mlx-lm: '--mask-prompt' ignores prompt and computes loss for just the completion, 'only supported for chat and completion datasets' (for chat, the final message is the completion). This applies to LoRA fine-tunes too — masking is orthogonal to LoRA vs full-tuning.",
        "sources": [
          "https://huggingface.co/docs/trl/sft_trainer",
          "https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md"
        ]
      },
      {
        "claim": "Surprisingly few examples are needed to teach tool-call format/behavior — minimum 10, recommended starting point ~50 well-crafted demonstrations, with measurable improvements from 50-100 examples — though the right number is use-case dependent and diversity/quality dominate raw count.",
        "confidence": "high",
        "vote": "unanimous (claim 1:3-0)",
        "evidence": "OpenAI: 'The minimum number of examples you can provide for fine-tuning is 10 ... We recommend starting with 50 well-crafted demonstrations ... We see improvements from fine-tuning on 50-100 examples, but the right number for you varies greatly.' Scope caveat: this is OpenAI hosted-SFT guidance for GPT models, not LoRA/QLoRA or MLX specifically, but it is the most concrete cited numeric anchor and corroborated by independent secondary sources (Parlance Labs, eesel.ai).",
        "sources": [
          "https://platform.openai.com/docs/guides/supervised-fine-tuning"
        ]
      },
      {
        "claim": "Diversity is the highest-leverage quality dimension: optimizing general-purpose diversity metrics across BOTH user-query phrasing (linguistic diversity) AND function-argument coverage yields +7.4% BFCL accuracy vs comparable synthetic-data baselines. These two axes (request phrasing, argument-value coverage) are under-explored relative to function/invocation/turn diversity, making them concrete levers for a corpus.",
        "confidence": "high",
        "vote": "unanimous (claims 5:3-0, 6:3-0)",
        "evidence": "arXiv 2601.17829 (TII/Technion, Jan 2026): 'we achieve an 7.4% increase in accuracy on the BFCL benchmark compared to similar counterparts' and 'Prior work emphasizes diversity in functions, invocation patterns, and interaction turns, yet linguistic diversity of requests and coverage of arguments remain underexplored.' Caveat: single self-reported benchmark, no independent replication; headline advantage is diversity/OOD generalization while keeping comparable correctness.",
        "sources": [
          "https://arxiv.org/abs/2601.17829"
        ]
      },
      {
        "claim": "Include NEGATIVE / error-recovery examples (failed tool call -> interpret feedback -> retry). Small models otherwise fall into repetitive invalid re-invocations after an error because standard training does not teach recovery. Note that purely STATIC pre-collected error-correction data degrades over training as it mismatches the policy's evolving failure modes — on-policy corrective supervision (Fission-GRPO, RL) is stronger, but error-recovery traces in SFT remain a recommended ingredient.",
        "confidence": "high",
        "vote": "unanimous (claims 7:3-0, 8:3-0)",
        "evidence": "arXiv 2601.15625 (Fission-GRPO): 'after a tool-call error, smaller models often fall into repetitive invalid re-invocations instead of interpreting the feedback and recovering ... current training paradigms do not explicitly teach models how to recover.' Same paper: 'pre-collected error-correction datasets become mismatched to the policy's evolving failure modes' (Qwen3-8B: +5.7% error recovery, 42.75%->46.75% accuracy with on-policy method). Corroborated by arXiv 2509.18847 and 2412.15495. Caveat: the static-dataset critique is the paper's motivating argument; SFT error-recovery examples still help format/behavior even if RL is stronger for robustness.",
        "sources": [
          "https://arxiv.org/pdf/2601.15625",
          "https://arxiv.org/abs/2509.18847",
          "https://arxiv.org/abs/2412.15495"
        ]
      },
      {
        "claim": "mlx-lm applies the model's Hugging Face chat template automatically for the chat, tools, and completions formats, so tool/tool_call rendering follows the model's own template (falling back to a HF default if the model lacks one). You do not hand-render the tool block.",
        "confidence": "high",
        "vote": "unanimous (claim 12:3-0)",
        "evidence": "mlx-lm LORA.md: 'In general, for the chat, tools and completions formats, Hugging Face chat templates are used.' Edge case: some models ship without a chat_template when training tools (ml-explore/mlx-examples #1243), but the doc covers this via HF default fallback.",
        "sources": [
          "https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md"
        ]
      },
      {
        "claim": "For long-context (32-64k) MLX LoRA tuning you MUST override max_seq_length — the official lora_config.yaml and the TrainingArgs code default are both 2048, and over-length samples are silently SLICED (truncated) to that length, not skipped, so the out-of-box default destroys long-context data.",
        "confidence": "high",
        "vote": "unanimous (claim 13:3-0)",
        "evidence": "mlx_lm/examples/lora_config.yaml: 'max_seq_length: 2048'; trainer.py TrainingArgs has max_seq_length default 2048; iterate_batches warns 'Some sequences are longer than {max_seq_length}... will be truncated' and executes truncated_length = min(lengths[j], max_seq_length) then slices. 2048 << 32k-64k.",
        "sources": [
          "https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/examples/lora_config.yaml",
          "https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md"
        ]
      },
      {
        "claim": "The default mlx-lm LoRA hyperparameters are rank 8, scale 20.0, dropout 0.0, applied only to attention q_proj and v_proj for the last num_layers layers. Important nuance: mlx-lm 'scale' is NOT alpha — scale = alpha/rank, so scale=20.0 with rank 8 implies alpha=160. For a 30B MoE long-context tune, expect to raise lora layers/rank and lower batch size to fit memory.",
        "confidence": "high",
        "vote": "majority (claim 14:2-1)",
        "evidence": "lora_config.yaml verbatim: keys: [\"self_attn.q_proj\",\"self_attn.v_proj\"], rank: 8, scale: 20.0, dropout: 0.0, with comment 'applied for the last lora_layers'. scale != alpha confirmed by mlx-examples #982 and #1186 (scale = alpha/rank). The dissent concerned only the loose '(alpha)' label; the numeric values are correct.",
        "sources": [
          "https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/examples/lora_config.yaml"
        ]
      },
      {
        "claim": "Sequence packing is the key throughput/memory technique for long-context SFT: it eliminates wasted padding compute (2x speedup for BERT phase-2 pretraining; 60-85% training-time reductions on large models, e.g. LLaMA-3-70B on 1.2M examples dropped ~184,709s -> ~27,426s). Packing benefits grow with model size (LLaMA-3-70B gained ~4.4 pts on WildChat vs ~1 pt for 8B), so it matters most for a 30B-class model.",
        "confidence": "high",
        "vote": "unanimous (claims 15:3-0, 16:3-0, 17:3-0)",
        "evidence": "arXiv 2410.08081v2: WildChat LLaMA-3-8B padding 49.58 vs greedy packing 50.6 (+1.02); 70B 61.50 vs 65.92 (+4.42); 70B 1.2M-set time 184,709s->27,426s (85.2%); 'Packing greatly reduces training time, making it possible to fine-tune large models on large datasets.' arXiv 2107.02027 (Graphcore): '2x speedup for phase 2 pre-training in BERT' (padding was ~50% of tokens at length 512).",
        "sources": [
          "https://arxiv.org/html/2410.08081v2",
          "https://arxiv.org/pdf/2107.02027"
        ]
      },
      {
        "claim": "Packing must use document-boundary attention masking (block-diagonal / varlen masks): tokens from different packed sequences must NOT attend across boundaries. This avoids cross-contamination, achieves mathematical equivalence to the unpacked model, and improves BOTH downstream performance and training throughput.",
        "confidence": "high",
        "vote": "consensus (claims 18:3-0, 21:2-1)",
        "evidence": "arXiv 2107.02027: 'tokens from different sequences within a pack should not be able to attend to each other' via block-diagonal masking, achieving 'mathematical equivalence between the original and packed models.' arXiv 2410.02660v4: 'we do not allow the attention to cross the document boundaries' — Table 20 shows improved long+short benchmarks; 'Disabling cross-document attention can also result in higher training throughput' (~11.7% with minibatch reordering, via FlashAttention-2 varlen). Corroborated by arXiv 2404.10830, FlashMask (ICLR 2025), 2407.09105. Caveat: throughput gain shrinks for long, low-variance sequence-length datasets.",
        "sources": [
          "https://arxiv.org/pdf/2107.02027",
          "https://arxiv.org/html/2410.02660v4",
          "https://arxiv.org/abs/2404.10830"
        ]
      },
      {
        "claim": "For long-context continued/fine-tuning, mix ~60% long data + 40% short data for best average performance; training only on long data hurts. Practical packing recipe: randomly sample and concatenate documents/conversations into fixed 64K chunks, truncating the last document per chunk (for SFT the truncated remainder is discarded; for short-data pretraining it is recycled into the next chunk).",
        "confidence": "high",
        "vote": "consensus (claims 19:3-0, 20:2-1)",
        "evidence": "arXiv 2410.02660v4 (ProLong, ACL 2025): 'The best average performance is achieved at 60% long data and 40% short data'; 'we randomly sample and concatenate the documents or conversations into 64K chunks. The last document for each chunk is truncated.' Caveats: validated at 8B (Llama-3-8B) on HELMET, measured after SFT; optimal ratio is task-dependent — treat 60/40 as a strong default, not a universal law. The 64K-chunk packing is described for short/SFT data specifically.",
        "sources": [
          "https://arxiv.org/html/2410.02660v4"
        ]
      },
      {
        "claim": "For realistic synthetic discovery/requirements interviewer turns, use the funnel technique (broad open-ended questions first, narrowing to specific closed questions later) and prefer open-ended experience-eliciting prompts over yes/no. Open-ended questions surface unanticipated info ('you don't know what you don't know'); closed questions are for clarification or quantitative analysis.",
        "confidence": "high",
        "vote": "unanimous (claims 22:3-0, 23:3-0, 24:3-0)",
        "evidence": "NN/g: 'the funnel technique, where the session or followup questions begin with broad, open-ended questions before introducing specific, closed questions'; open-ended questions 'allow you to find more than you anticipate'; closed questions used 'to gather additional small details, gain clarification, or when you want to analyze responses quantitatively.' Recommended phrasings: 'Walk me through a typical day for you', 'Tell me about the last time you [did something]', 'ask about specific events rather than about general processes.' NN/g is the canonical UX-research authority; methodology is evergreen.",
        "sources": [
          "https://www.nngroup.com/articles/open-ended-questions/",
          "https://www.nngroup.com/articles/user-interviews/",
          "https://www.nngroup.com/articles/the-funnel-technique-in-qualitative-user-research/"
        ]
      }
    ],
    "caveats": "Coverage is uneven across the four parts of the research question. Parts A and B (tool-calling training data, loss masking, packing, long-context mixing, mlx-lm specifics) are strongly supported by primary sources (OpenAI/Azure docs, HF TRL docs, the canonical ml-explore/mlx-lm repo, and peer-reviewed/preprint papers with verbatim quotes). Part C (discovery interview technique) is well-covered by NN/g primary sources but ONLY for the funnel/open-vs-closed dimensions — the verified claims do NOT cover 5-Whys, MoSCoW prioritization, probing for unstated assumptions, or one-question-at-a-time pacing, so those parts of the question remain unsourced here. Part D (user stories & story mapping: INVEST, As-a/I-want/so-that template, Gherkin/Given-When-Then acceptance criteria, epics->stories->sub-stories, story mapping backbone, story->task decomposition) has ZERO verified claims — it was not substantiated by the surviving evidence and should be treated as an open gap. Time-sensitivity: mlx-lm details are from the live main branch (current as of 2026-06) and can change between releases; the +7.4% BFCL and Fission-GRPO results are single 2026 preprints without independent replication; the 60/40 long/short mix was validated at 8B on the HELMET suite, not on a 30B MoE. The OpenAI 50-100 example guidance is for hosted GPT SFT, not directly measured for LoRA/QLoRA or MLX, though it is the most concrete numeric anchor available. One value-label nuance: mlx-lm 'scale' (20.0) is alpha/rank, not alpha. Two claims (0, 14, 20, 21) carried a 2-1 dissent but the dissent targeted wording/scoping (the word 'must', the '(alpha)' label, blanket generality), not the substantive facts.",
    "openQuestions": [
      "What are the concrete patterns/templates for USER STORIES and STORY MAPPING (INVEST criteria, As-a/I-want/so-that, Gherkin Given-When-Then acceptance criteria, epic->story->task decomposition, the story-map backbone)? No verified claims covered Part D — this needs a separate research pass.",
      "For the additional discovery-interview techniques named in the question — 5 Whys, MoSCoW prioritization, probing for edge/error cases and unstated assumptions, one-question-at-a-time pacing — what are authoritative sources and concrete examples? Only the funnel/open-vs-closed dimension was verified.",
      "What are the empirically validated example counts and long/short ratios specifically for LoRA/QLoRA tool-calling fine-tunes (vs OpenAI hosted SFT) and specifically on a 30B MoE — does the 50-100 floor and 60/40 mix transfer, or do MoE/long-context dynamics change the recommendation?",
      "What are concrete mlx-lm config values (max_seq_length, lora layers, rank, batch size, gradient checkpointing) that actually fit a 30B MoE at 32-64k context within Apple-Silicon unified-memory limits, and does mlx-lm implement document-boundary attention masking during packing (or only naive concatenation)?"
    ],
    "refuted": [],
    "sources": [
      {
        "url": "https://platform.openai.com/docs/guides/supervised-fine-tuning",
        "quality": "primary",
        "angle": "tool-calling fine-tune data format & loss masking (primary technical)",
        "claimCount": 5
      },
      {
        "url": "https://huggingface.co/docs/trl/sft_trainer",
        "quality": "primary",
        "angle": "tool-calling fine-tune data format & loss masking (primary technical)",
        "claimCount": 5
      },
      {
        "url": "https://yonigottesman.github.io/2024/05/13/mask-user-tokens.html",
        "quality": "blog",
        "angle": "tool-calling fine-tune data format & loss masking (primary technical)",
        "claimCount": 5
      },
      {
        "url": "https://huggingface.co/learn/cookbook/function_calling_fine_tuning_llms_on_xlam",
        "quality": "secondary",
        "angle": "tool-calling fine-tune data format & loss masking (primary technical)",
        "claimCount": 5
      },
      {
        "url": "https://www.stephendiehl.com/posts/fine_tuning_tools/",
        "quality": "blog",
        "angle": "tool-calling fine-tune data format & loss masking (primary technical)",
        "claimCount": 5
      },
      {
        "url": "https://particula.tech/blog/how-much-data-fine-tune-llm",
        "quality": "blog",
        "angle": "LoRA/QLoRA dataset size, diversity & error-recovery examples (empirical numbers)",
        "claimCount": 5
      },
      {
        "url": "https://latitude.so/blog/dataset-size-impacts-llm-fine-tuning",
        "quality": "blog",
        "angle": "LoRA/QLoRA dataset size, diversity & error-recovery examples (empirical numbers)",
        "claimCount": 5
      },
      {
        "url": "https://arxiv.org/abs/2601.17829",
        "quality": "primary",
        "angle": "LoRA/QLoRA dataset size, diversity & error-recovery examples (empirical numbers)",
        "claimCount": 4
      },
      {
        "url": "https://arxiv.org/pdf/2601.15625",
        "quality": "primary",
        "angle": "LoRA/QLoRA dataset size, diversity & error-recovery examples (empirical numbers)",
        "claimCount": 3
      },
      {
        "url": "https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md",
        "quality": "primary",
        "angle": "mlx-lm LoRA JSONL chat format & long-context config (implementation)",
        "claimCount": 5
      },
      {
        "url": "https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/examples/lora_config.yaml",
        "quality": "primary",
        "angle": "mlx-lm LoRA JSONL chat format & long-context config (implementation)",
        "claimCount": 5
      },
      {
        "url": "https://medium.com/@levchevajoana/fine-tuning-a-model-for-function-calling-with-mlx-lm-d00d587e2559",
        "quality": "blog",
        "angle": "mlx-lm LoRA JSONL chat format & long-context config (implementation)",
        "claimCount": 5
      },
      {
        "url": "https://arxiv.org/html/2410.08081v2",
        "quality": "primary",
        "angle": "long-context training: sequence packing, truncation, decontamination",
        "claimCount": 5
      },
      {
        "url": "https://arxiv.org/pdf/2107.02027",
        "quality": "primary",
        "angle": "long-context training: sequence packing, truncation, decontamination",
        "claimCount": 4
      },
      {
        "url": "https://arxiv.org/html/2410.02660v4",
        "quality": "primary",
        "angle": "long-context training: sequence packing, truncation, decontamination",
        "claimCount": 5
      },
      {
        "url": "https://developer.nvidia.com/blog/mastering-llm-techniques-data-preprocessing/",
        "quality": "secondary",
        "angle": "long-context training: sequence packing, truncation, decontamination",
        "claimCount": 5
      },
      {
        "url": "https://khooanxian.medium.com/summary-how-abilities-in-llms-are-affected-by-supervised-fine-tuning-data-composition-82a574474099",
        "quality": "secondary",
        "angle": "long-context training: sequence packing, truncation, decontamination",
        "claimCount": 5
      },
      {
        "url": "https://www.nngroup.com/articles/open-ended-questions/",
        "quality": "primary",
        "angle": "requirements discovery interview technique (PM/UX/consulting)",
        "claimCount": 5
      },
      {
        "url": "https://www.nngroup.com/articles/user-interviews/",
        "quality": "primary",
        "angle": "requirements discovery interview technique (PM/UX/consulting)",
        "claimCount": 5
      },
      {
        "url": "https://www.nngroup.com/articles/interview-guide/",
        "quality": "primary",
        "angle": "requirements discovery interview technique (PM/UX/consulting)",
        "claimCount": 5
      },
      {
        "url": "https://www.softwaretestinghelp.com/requirements-elicitation-techniques/",
        "quality": "blog",
        "angle": "requirements discovery interview technique (PM/UX/consulting)",
        "claimCount": 5
      },
      {
        "url": "https://maze.co/guides/user-interviews/questions/",
        "quality": "blog",
        "angle": "requirements discovery interview technique (PM/UX/consulting)",
        "claimCount": 5
      },
      {
        "url": "https://nextgenanalysts.co.uk/complete-guide-to-eliciting-requirements-in-agile-projects/",
        "quality": "blog",
        "angle": "requirements discovery interview technique (PM/UX/consulting)",
        "claimCount": 5
      },
      {
        "url": "https://jpattonassociates.com/the-new-backlog/",
        "quality": "primary",
        "angle": "user stories, INVEST, Gherkin acceptance criteria & story mapping",
        "claimCount": 5
      },
      {
        "url": "https://www.nngroup.com/articles/user-story-mapping/",
        "quality": "secondary",
        "angle": "user stories, INVEST, Gherkin acceptance criteria & story mapping",
        "claimCount": 4
      },
      {
        "url": "https://jpattonassociates.com/wp-content/uploads/2015/03/story_mapping.pdf",
        "quality": "primary",
        "angle": "user stories, INVEST, Gherkin acceptance criteria & story mapping",
        "claimCount": 5
      },
      {
        "url": "https://www.altexsoft.com/blog/acceptance-criteria-purposes-formats-and-best-practices/",
        "quality": "blog",
        "angle": "user stories, INVEST, Gherkin acceptance criteria & story mapping",
        "claimCount": 5
      },
      {
        "url": "https://www.businessanalysisexperts.com/gherkin-user-stories-given-when-then-examples/",
        "quality": "blog",
        "angle": "user stories, INVEST, Gherkin acceptance criteria & story mapping",
        "claimCount": 5
      },
      {
        "url": "https://mpug.com/the-big-picture-with-story-map-in-agile-development",
        "quality": "blog",
        "angle": "user stories, INVEST, Gherkin acceptance criteria & story mapping",
        "claimCount": 5
      }
    ],
    "stats": {
      "angles": 6,
      "sourcesFetched": 29,
      "claimsExtracted": 140,
      "claimsVerified": 25,
      "confirmed": 25,
      "killed": 0,
      "afterSynthesis": 12,
      "urlDupes": 0,
      "budgetDropped": 7,
      "agentCalls": 112
    }
  }
}