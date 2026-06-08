{
  "summary": "Deep research harness — fan-out web searches, fetch sources, adversarially verify claims, synthesize a cited report.",
  "agentCount": 110,
  "logs": [
    "Q: What are the best, state-of-the-art methods (2023-2026) for generating HIGH-QUAL…",
    "Decomposed into 6 angles: primary synthesis methods, tool-calling / function-call datasets, model collapse & low-entropy memorization, quality control & filtering, data scale, mix, and diminishing returns, intent classification / taxonomy inference data",
    "intent classification / taxonomy inference data: 6 results",
    "primary synthesis methods: 6 results",
    "quality control & filtering: 6 results",
    "quality control & filtering: 4 novel (2 filtered)",
    "tool-calling / function-call datasets: 6 results",
    "tool-calling / function-call datasets: 4 novel (2 filtered)",
    "model collapse & low-entropy memorization: 6 results",
    "model collapse & low-entropy memorization: 4 novel (2 filtered)",
    "data scale, mix, and diminishing returns: 6 results",
    "data scale, mix, and diminishing returns: 3 novel (3 filtered)",
    "Fetched 27 sources → 124 claims → verifying top 25",
    "\"WizardLM fine-tuned on Evol-Instruct data is prefe…\": 3-0 ✓",
    "\"Self-Instruct fine-tuning yielded a 33% absolute i…\": 3-0 ✓",
    "\"Evol-Instruct uses an LLM to automatically rewrite…\": 3-0 ✓",
    "\"Self-Instruct is a pipeline that bootstraps synthe…\": 3-0 ✓",
    "\"Persona-Hub introduces a persona-driven synthetic …\": 3-0 ✓",
    "\"Human evaluation found that the AI-evolved instruc…\": 3-0 ✓",
    "\"The authors curated Persona Hub, a collection of 1…\": 3-0 ✓",
    "\"Genetic-Instruct synthesizes large-scale coding in…\": 3-0 ✓",
    "\"The persona-driven approach is demonstrated to syn…\": 0-3 ✗",
    "\"The 'attributed grounding' method works in two pha…\": 3-0 ✓",
    "\"Grounding generation in real web documents plus se…\": 3-0 ✓",
    "\"The method scales to over 7.5 million synthetic co…\": 3-0 ✓",
    "\"APIGen verifies every generated function-calling d…\": 3-0 ✓",
    "\"Tag-Evol improves on Evol-Instruct by injecting di…\": 3-0 ✓",
    "\"SynthQuestions ranks top in data diversity among s…\": 2-1 ✓",
    "\"The released APIGen dataset contains 60,000 high-q…\": 3-0 ✓",
    "\"ToolACE uses an automated agentic pipeline whose A…\": 3-0 ✓",
    "\"Tag-Evol outperforms Evol-Instruct baselines by 2-…\": 3-0 ✓",
    "\"ToolACE generates dialogs through interplay among …\": 3-0 ✓",
    "\"Models trained on APIGen-curated data reach state-…\": 3-0 ✓",
    "\"ToolACE ensures data accuracy with a dual-layer ve…\": 3-0 ✓",
    "\"A model with only 8B parameters trained on ToolACE…\": 3-0 ✓",
    "\"xLAM uses the APIGen synthesis framework to genera…\": 3-0 ✓",
    "\"Diversity is engineered via prompt-format augmenta…\": 3-0 ✓",
    "\"The synthetic function-calling dataset was built f…\": 1-2 ✗",
    "Verify done: 25 claims → 23 confirmed, 2 killed"
  ],
  "result": {
    "question": "What are the best, state-of-the-art methods (2023-2026) for generating HIGH-QUALITY, HIGH-DIVERSITY synthetic training data for fine-tuning an LLM agent — specifically a tool-calling INTERVIEW/requirements agent that must (a) map free-text user descriptions to the correct category/tags via INFERENCE (e.g. \"I want to sell lemonade\" → industry = Food & Beverage), and (b) emit correct multi-step function calls. I need concrete, cited, actionable techniques I can implement in a Python generator and/or via LLM distillation:\n\n(1) SYNTHESIS METHODS: Self-Instruct, Evol-Instruct/WizardLM (instruction evolution — deepening, broadening, increasing complexity), Persona-Hub / persona-driven generation (using diverse personas/seeds to maximize coverage), Genie, back-translation, and distillation from a strong teacher model. For each: how it works, what diversity it buys, and when to use it.\n\n(2) AVOIDING THE \"TRAINS TOO FAST / MEMORIZES\" FAILURE: why templated/low-entropy synthetic data causes a model to converge in very few steps and over-memorize (loss → near zero fast) without generalizing; how to raise effective ENTROPY and naturalness (paraphrase, varied phrasing/register/length, noise, real-world seeds); the role of n-gram/embedding diversity metrics; mode collapse in synthetic data and model-collapse risk from training on synthetic data.\n\n(3) QUALITY CONTROL: rejection sampling / filtering (reward models, LLM-as-judge, execution/verification filtering for tool calls — does the call validate against the schema?), dedup (MinHash/embedding near-dup removal), difficulty balancing, and decontamination. How much filtering vs raw generation.\n\n(4) TEACHING INFERENCE/MAPPING specifically: how to build data that teaches a model to map varied natural-language inputs to a fixed taxonomy (intent classification / slot filling / entity-to-category), including hard/ambiguous cases, negatives, and \"ask a clarifying question when ambiguous\" behavior.\n\n(5) HOW MUCH DATA + MIX: for teaching tool-call FORMAT vs BEHAVIOR vs broad generalization, recommended scale (is 100k-1M synthetic examples useful or does it plateau?), the diminishing returns curve, and how to mix synthetic with a little real/human data. Cite numbers.\n\nPrefer primary sources: Self-Instruct (arXiv 2212.10560), WizardLM/Evol-Instruct, Microsoft Orca, Persona-Hub (arXiv 2406.20094), the \"model collapse\" Nature paper, NVIDIA Nemotron data papers, and tool-calling dataset papers (xLAM/APIGen, Glaive, ToolACE). Give specific, implementable recommendations.",
    "summary": "Recipe for high-quality high-diversity synthetic data for a tool-calling interview agent.",
    "caveats": "All 23 claims are primary sourced and 22 had unanimous votes. BFCL SOTA results are time sensitive 2024 snapshots and several results are self reported. Personas do not reliably raise lexical diversity. No surviving claims covered model collapse, Orca, Genie, back translation, or Nemotron, so data mix and collapse numbers are under evidenced.",
    "findings": [
      {
        "claim": "Layer synthesis methods Self-Instruct base then Evol-Instruct then Tag-Evol then Genetic-Instruct for rising complexity from a small seed.",
        "confidence": "high",
        "evidence": "arXiv 2212.10560 with 2304.12244 2505.24165 2407.21077.",
        "sources": [
          "arXiv 2212.10560"
        ]
      },
      {
        "claim": "Diversity persona seeds plus real-document grounding is the biggest lever measure Vendi and MTLD and apply xLAM entropy augmentations.",
        "confidence": "high",
        "evidence": "arXiv 2506.03968 with 2406.20094 2409.03215 2505.17390.",
        "sources": [
          "arXiv 2506.03968"
        ]
      },
      {
        "claim": "Tool-call quality control by execution verification beats volume but leaderboard rank overstates robustness under paraphrase.",
        "confidence": "high",
        "evidence": "arXiv 2406.18518 with 2409.00920 2509.26553.",
        "sources": [
          "arXiv 2406.18518"
        ]
      }
    ],
    "refuted": [
      {
        "claim": "The persona-driven approach is demonstrated to synthesize tool/function-calling data and user instructions at scale, making it directly applicable to generating tool-calling and interview/requirements agent training data.",
        "vote": "0-3",
        "source": "https://arxiv.org/abs/2406.20094"
      },
      {
        "claim": "The synthetic function-calling dataset was built from over 3,673 executable APIs across 21 categories, yielding 60,000 high-quality verified examples generated by strong open models (DeepSeek-V2-Chat, Mixtral-8x22B-Inst).",
        "vote": "1-2",
        "source": "https://arxiv.org/abs/2409.03215"
      }
    ],
    "sources": [
      {
        "url": "https://arxiv.org/abs/2212.10560",
        "quality": "primary",
        "angle": "primary synthesis methods",
        "claimCount": 4
      },
      {
        "url": "https://arxiv.org/abs/2304.12244",
        "quality": "primary",
        "angle": "primary synthesis methods",
        "claimCount": 4
      },
      {
        "url": "https://arxiv.org/abs/2406.20094",
        "quality": "primary",
        "angle": "primary synthesis methods",
        "claimCount": 4
      },
      {
        "url": "https://arxiv.org/pdf/2506.03968",
        "quality": "primary",
        "angle": "primary synthesis methods",
        "claimCount": 5
      },
      {
        "url": "https://arxiv.org/pdf/2407.21077",
        "quality": "primary",
        "angle": "primary synthesis methods",
        "claimCount": 5
      },
      {
        "url": "https://arxiv.org/pdf/2505.24165",
        "quality": "primary",
        "angle": "primary synthesis methods",
        "claimCount": 5
      },
      {
        "url": "https://arxiv.org/abs/2406.18518",
        "quality": "primary",
        "angle": "tool-calling / function-call datasets",
        "claimCount": 5
      },
      {
        "url": "https://arxiv.org/abs/2409.00920",
        "quality": "primary",
        "angle": "tool-calling / function-call datasets",
        "claimCount": 4
      },
      {
        "url": "https://arxiv.org/abs/2409.03215",
        "quality": "primary",
        "angle": "tool-calling / function-call datasets",
        "claimCount": 5
      },
      {
        "url": "https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k",
        "quality": "primary",
        "angle": "tool-calling / function-call datasets",
        "claimCount": 5
      },
      {
        "url": "https://www.nature.com/articles/s41586-024-07566-y",
        "quality": "primary",
        "angle": "model collapse & low-entropy memorization",
        "claimCount": 5
      },
      {
        "url": "https://arxiv.org/abs/2410.15226",
        "quality": "primary",
        "angle": "model collapse & low-entropy memorization",
        "claimCount": 4
      },
      {
        "url": "https://arxiv.org/html/2509.16499v1",
        "quality": "primary",
        "angle": "model collapse & low-entropy memorization",
        "claimCount": 5
      },
      {
        "url": "https://amitness.com/posts/diversity-evals/",
        "quality": "blog",
        "angle": "model collapse & low-entropy memorization",
        "claimCount": 5
      },
      {
        "url": "https://arxiv.org/abs/2406.11704",
        "quality": "primary",
        "angle": "quality control & filtering",
        "claimCount": 5
      },
      {
        "url": "https://arxiv.org/pdf/2406.18518",
        "quality": "primary",
        "angle": "quality control & filtering",
        "claimCount": 4
      },
      {
        "url": "https://developer.nvidia.com/blog/build-custom-reasoning-models-with-advanced-open-post-training-datasets/",
        "quality": "primary",
        "angle": "quality control & filtering",
        "claimCount": 5
      },
      {
        "url": "https://developer.nvidia.com/blog/mastering-llm-techniques-data-preprocessing/",
        "quality": "primary",
        "angle": "quality control & filtering",
        "claimCount": 5
      },
      {
        "url": "https://openreview.net/pdf?id=KBMOKmX2he",
        "quality": "primary",
        "angle": "data scale, mix, and diminishing returns",
        "claimCount": 5
      },
      {
        "url": "https://arxiv.org/html/2503.19551v3",
        "quality": "primary",
        "angle": "data scale, mix, and diminishing returns",
        "claimCount": 5
      },
      {
        "url": "https://arxiv.org/abs/2502.04194",
        "quality": "primary",
        "angle": "data scale, mix, and diminishing returns",
        "claimCount": 5
      },
      {
        "url": "https://aclanthology.org/2024.acl-long.578.pdf",
        "quality": "primary",
        "angle": "intent classification / taxonomy inference data",
        "claimCount": 5
      },
      {
        "url": "https://proceedings.iclr.cc/paper_files/paper/2025/file/97e2df4bb8b2f1913657344a693166a2-Paper-Conference.pdf",
        "quality": "primary",
        "angle": "intent classification / taxonomy inference data",
        "claimCount": 5
      },
      {
        "url": "https://arxiv.org/pdf/2601.11234",
        "quality": "primary",
        "angle": "intent classification / taxonomy inference data",
        "claimCount": 5
      },
      {
        "url": "https://www.tandfonline.com/doi/full/10.1080/08839514.2024.2414483",
        "quality": "primary",
        "angle": "intent classification / taxonomy inference data",
        "claimCount": 2
      },
      {
        "url": "https://arxiv.org/html/2510.08517v1",
        "quality": "primary",
        "angle": "intent classification / taxonomy inference data",
        "claimCount": 5
      },
      {
        "url": "https://arxiv.org/pdf/2504.12113",
        "quality": "primary",
        "angle": "intent classification / taxonomy inference data",
        "claimCount": 3
      }
    ],
    "stats": {
      "angles": 6,
      "sourcesFetched": 27,
      "claimsExtracted": 124,
      "claimsVerified": 25,
      "confirmed": 23,
      "killed": 2,
      "afterSynthesis": 3,
      "urlDupes": 1,
      "budgetDropped": 8,
      "agentCalls": 110
    }
  }
}