# Interview Study Notes

## RAG & LLM Evaluation — Metrics & Measurement Methods

### Quantitative Metrics

| Metric | What it means | Fintech analogy |
|---|---|---|
| **Accuracy** | % of correct answers overall | Basic model performance |
| **Precision** | When model says "yes", how often correct? | Low false positives — flagging real fraud only |
| **Recall** | Of all actual "yes" cases, how many caught? | Low false negatives — catching all fraud |
| **F1 Score** | Harmonic mean of precision + recall | Use when both matter — fraud/risk models |
| **BLEU Score** | How closely generated text matches reference | Translation, summarization — less relevant for RAG |
| **Perplexity** | How "surprised" is the model — lower = more confident | Internal LLM metric, rarely used in RAG evals |
| **Response time** | Latency in ms | Same as API/pipeline SLAs |

### Quantitative Methods

- **A/B testing** — compare old RAG pipeline vs. new one; measure which scores higher on F1 or user preference
- **User feedback / task completion** — implicit signal: did user ask follow-up? did they copy the answer?
- **Edge case analysis** — % of unusual questions handled without hallucination or failure

### Qualitative Scales

- **Likert scale (1–5)** — human or LLM-as-judge rates the answer (e.g. faithfulness to source doc)
- **Expert rubrics** — structured scoring criteria; define exactly what a "5" looks like vs. a "3"

### How It All Fits for RAG (the Harness)

```
RAG Pipeline
     ↓
Run test questions → get answers
     ↓
Quantitative: Score with Ragas (faithfulness, relevancy, recall)
Qualitative:  LLM-as-judge rates coherence on a rubric
Operational:  Log response time, token usage
     ↓
A/B test: chunk size 256 vs 512 tokens → which scores better?
     ↓
This is your harness
```

**Key insight:** For RAG, **faithfulness** and **context recall** are your F1/precision equivalents —
they tell you if the model is using retrieved docs vs. hallucinating.

### Recommended Learning Path

```
Current RAG course (DeepLearning.AI) → Ragas/TruLens hands-on → LangSmith → W&B Eval course
```

---

<!-- Add new topics below as you study -->
