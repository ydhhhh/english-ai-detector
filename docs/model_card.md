# English AI Detector v2 model card

Model ID: `english-ai-detector-v2-practical`.

This model estimates whether English prose is human-written or AI-generated/substantively AI-rewritten. It also returns **exploratory source-family and nominal-version rankings conditional on AI**. Both source assertion switches are disabled: `accepted_family` and `accepted_model` are always `null` in this release.

The frozen binary system passed the preregistered point criteria on four necessary final cohorts and both required rewrite-origin views. This is evidence for those measured populations, not universal detection accuracy, a per-domain 1% false-positive guarantee, or authenticated model attribution. Local engineering checks are recorded separately from these statistical results.

## Supported use and decisions

The supported core is English natural-language prose with at least **128 original-text word-like units**, counted using Unicode regex `\b\w+\b`, not encoder subwords. The English guard samples limited positions and cannot identify every locally mixed-language passage. Shorter or unsupported-language inputs receive exploratory probabilities and an `inconclusive` decision.

The binary AI class includes direct generation, aligned generation and substantive machine rewriting, including **a human original rewritten by DIPPER**. An AI decision does not mean every word was originally written by a machine. The model does not estimate an AI-written word fraction, sentence-level origin, authorship identity or factual correctness.

The deployed rules use one fixed pair of thresholds across all cohorts:

- AI region: `P(AI) >= 0.7233173320180257`.
- Human region: `P(human) >= 0.9049688379295407`, where `P(human) = 1 - P(AI)`.
- A decision also requires the English/core-length guard and an exclusive directional result. All other cases are inconclusive; exactly `P(AI)=0.5` is inconclusive.

The gap between thresholds is intentional. A forced binary label at 0.5 is a separate diagnostic and is not the shipping decision policy. Probabilities depend on the calibration mixture and its class/domain distribution; they are not population-invariant authorship probabilities.

Full token streams are processed with overlapping windows and new-token-weighted logit aggregation; the tail is not silently dropped. Runtime increases with length. The local HTTP entry defaults to 200,000 characters, 8 MiB per upload, six requests/minute and one concurrent detection; it rejects excess input. CLI/Python capacity remains limited by available memory and time.

## Four inference components

| Component | Function | Neural parameters |
|---|---|---:|
| `minilm4_epoch1` | Standardized binary margin, weight 0.25 | 22,920,354 |
| `modern3_epoch2` | Standardized binary margin, weight 0.75 | 149,419,554 |
| `minilm_v3_epoch2` | Inherited family/25-leaf source probabilities | 22,920,354 |
| `statistical_v2` | Inherited within-family nominal-version refinement; no binary blend weight | Linear statistical model |

The 75/25 blend acts on standardized raw margins, not on two probability vectors. The new binary encoders' fine logits do not replace the independent old source encoder.

| Trained artifact | SHA256 |
|---|---|
| Mini4 epoch 1 | `e1877e1665c31a6070cfe0f0f6a5b2cea051df9382c04ebe9fb98b2dddb465e0` |
| Modern3 epoch 2 | `84bbb9e8efe59cbdc050940fa7f63bf4d5b493d27cea84fd7d6f68dbb7649f25` |
| Inherited Mini3 epoch 2 | `9d2ca9d251aeb2f2067478c12c564be863c478b91c579e4ea312f5170c04aabb` |
| Statistical v2 | `32a35bcfa2fc09a49d841bf2f2ad67f3d0d5fa926590c282f4dd14363c41fc45` |

The neural ancestors originate from [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/1110a243fdf4706b3f48f1d95db1a4f5529b4d41) and [ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base/tree/8949b909ec900327062f0ebf497f51aef5e6f0c8), whose publishers declare Apache-2.0. Fine-tuned artifacts also retain their documented training histories and publication notice; the source-code license is a separate scope.

## Calibration and experimental boundaries

The v2 adaptation set contains 60,000 records, human/AI 30,000 each: per class, 10,000 new and 20,000 replay records. Mini4 and Modern3 each trained for two epochs with fresh optimizers. Family/version loss weights were exactly zero without altering the labels or credibility masks. Selection compared four epoch candidates on six registered development cohorts with group-disjoint fit/selection halves. Final scores did not choose the blend or thresholds.

Binary calibration used a registered 8,000-row mixture: 6,000 fresh-source CAL records plus **2,000 previously observed historical WP CAL replay records**. It is not wholly fresh calibration. The supported-English core fitting subset contains 7,992 records, human 3,795 and AI 4,197. Eight unsupported records were excluded from fitting, not deleted from the final-test denominators.

The frozen binary map is `sigmoid(5.880817144501673 * blend_margin + 0.757325435191797)`. On its calibration fitting subset, the AI cutoff admitted 37/3,795 human-as-AI errors, and the human cutoff admitted 41/4,197 AI-as-human errors. The two 1% budgets are finite-sample **mixture** constraints, not per-source or population bounds.

Source hierarchy, refinement and temperatures are inherited unchanged from the older source calibration. Old family/model cutoffs are inactive; the source release gate and source protocol are `null`. Binary success does not certify the source rankings.

## Final evaluation: deployed behavior with abstentions

All 7,600 final records meet the 128-word-like core threshold. The four cohorts are separate measured populations; no pooled universal accuracy is claimed. AI recall and directional error rates below retain **every member of the true class**, including inconclusive and language-rejected records. Coverage uses all records.

Values in brackets are 95% percentile whole-connected-group bootstrap intervals: 500 replicates, seed 42, with fixed predictions and thresholds. Paired/triplet rows move together. The intervals exclude retraining, model-selection and calibration uncertainty.

| Final cohort | Human / AI; groups | AUROC [95% CI] | Applied-policy correct-AI recall [95% CI] | Human → AI [95% CI] |
|---|---|---|---|---|
| Main: unused MAGE components | 1,500 / 1,500; 3,000 | 0.997829 [0.996428, 0.998788] | 97.60% [96.88%, 98.24%] | 1.40% [0.88%, 2.02%] |
| Source pipeline: Natural Questions | 400 / 400; 400 | 0.999388 [0.998429, 0.999960] | 99.50% [98.75%, 100.00%] | 3.50% [1.75%, 5.38%] |
| Modern: MAGA/MGB with human controls | 1,000 / 1,000; 1,000 | 0.997897 [0.995859, 0.999439] | 97.80% [96.90%, 98.75%] | 0.70% [0.20%, 1.30%] |
| Paraphrase: RAID/DIPPER with human controls | 600 / 1,200; 600 | 0.995828 [0.993346, 0.997562] | 95.17% [93.87%, 96.25%] | 0.17% [0.00%, 0.50%] |

| Final cohort | Correct-human recall [95% CI] | AI → human [95% CI] | Inconclusive / all records [95% CI] |
|---|---|---|---|
| Main | 95.73% [94.61%, 96.76%] | 0.80% [0.39%, 1.30%] | 2.23% [1.67%, 2.77%] |
| Natural Questions | 87.75% [84.25%, 90.75%] | 0.00% [0.00%, 0.00%] | 4.63% [3.38%, 6.25%] |
| Modern | 95.50% [94.10%, 96.70%] | 0.90% [0.40%, 1.50%] | 2.55% [1.85%, 3.25%] |
| Paraphrase | 95.00% [93.17%, 96.67%] | 1.58% [0.96%, 2.38%] | 3.78% [2.94%, 4.72%] |

Acceptance required all preregistered point checks on each necessary cohort and rewrite view. It did **not** require every confidence bound to meet the point target. For example, main FPR passes its 2% point criterion but its interval extends above 2%; NQ FPR passes its 5% point criterion but its interval extends above 5%. Zero observed errors with a degenerate bootstrap interval do not prove zero population risk.

The modern cohort has five language-rejected records and NQ has two; they remain in the denominators. Modern forced-score AI recall at the AI cutoff is 98.00%, while actual applied-policy recall is **97.80%**. Reporting only the forced score would overstate applied-policy recall. At the separate forced 0.5 threshold, balanced accuracy is 98.13% / 97.38% / 98.75% / 97.83%; these point diagnostics have no separately computed bootstrap intervals in the original reports.

## Two rewrite views and weaker slices

The paraphrase cohort contains clean human originals, DIPPER rewrites of human originals, and DIPPER rewrites of AI originals. The two required views share the **same 600 human controls**, overlap their 1,800-row parent cohort, and are neither additive nor independent extra tests.

| Required view | Human / AI | Applied-policy correct-AI recall [95% CI] | AI → human [95% CI] | Inconclusive [95% CI] |
|---|---|---|---|---|
| Human-origin DIPPER rewrite | 600 / 600 | 92.33% [90.17%, 94.25%] | 2.83% [1.67%, 4.17%] | 4.83% [3.67%, 6.04%] |
| AI-origin DIPPER rewrite | 600 / 600 | 98.00% [96.75%, 99.17%] | 0.33% [0.00%, 0.83%] | 3.25% [2.25%, 4.42%] |

Post-final descriptive breakdowns identify weaker cells: human-origin DIPPER **news 272/300 detected, 90.67% [87.00%, 93.67%]**; Ministral-8B-Instruct-2410 **59/64, 92.19% [84.38%, 96.88%]**; Qwen3-plus **60/64, 93.75% [87.50%, 98.44%]**. The held-out MAGA Gemini-2.0-flash identifier has 292/300 detections, 97.33% [95.33%, 99.00%], four AI-as-human errors and four inconclusive AI records. Most smaller generator slices contain only 62–64 AI records. Generator/configuration/domain slices overlap, share controls, and are descriptive rather than randomized causal comparisons. Observed 100% slices are not perfect-future-performance guarantees.

This evidence covers the published English DIPPER setup, not arbitrary human editing, translation, every paraphraser or an AI–human collaboration proportion.

## Source rankings are not source authentication

The seven families are Claude, GPT, Gemini, Qwen, GLM, Kimi and Other. The 25 leaves comprise 18 nominal model labels, six within-family `other_version` buckets and `other_family` (family `Other`). Leaves sum to their family mass; all family/leaf probabilities are conditional on AI, including rankings returned for a human or inconclusive input.

The nominal catalog is:

- Claude: `claude-fable-5`, `claude-opus-4-6`, `claude-opus-4-8`, `claude-opus-5`, `claude-sonnet-5`.
- GPT: `gpt-5.5`, `gpt-5.6-luna`, `gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-6-astra`.
- Gemini: `gemini-3.1-pro-preview`, `gemini-3.6-flash`, `gemini-3.7-flash`, `gemini-3.8-flash`.
- Qwen: `qwen3.8-max`; GLM: `glm-5.3`, `glm-5.3-flash`; Kimi: `kimi-k3`.

These names are publisher/API assertions and may be service aliases. They are not independent authentication of the weights actually served. `Other` buckets are supervised observed catalog complements, not calibrated probabilities covering every possible future model. The original generator and the last editor of rewritten text are distinct. Source assertion coverage is zero in this release; source accuracy is therefore undefined, not 100%.

## Data, reproduction and rights limits

Training and ancestors include MAGE, HC3, community/competition material and WP adaptation. WikiHow raw/replay was excluded from v2, but the warm-start weights retain earlier exposure; no unlearning or clean-from-scratch claim is made. NQ and the MAGA Gemini identifier were held out from v2 adaptation train/dev/CAL, not certified absent from all ancestors or general pretraining. Observable native-parent, exact/hash, title and approximate lexical components were isolated; hidden parents, semantic overlap and pretraining overlap cannot be ruled out. The v1 graph-independence erratum remains disclosed in the data documentation, and old final sets are historical diagnostics, not new v2 blind evidence.

The project does not release corpus bodies or automatically resolve upstream data rights. Source code is Apache-2.0 with preserved MIT history. Model publication uses a separate Research-Use Model Notice; it is not a blanket Apache grant for the weights or proof of all third-party redistribution rights. See [data sources](https://github.com/ydhhhh/english-ai-detector/blob/main/docs/data_sources.md), [training](https://github.com/ydhhhh/english-ai-detector/blob/main/docs/training.md) and [licensing](https://github.com/ydhhhh/english-ai-detector/blob/main/docs/licensing.md).

Dynamic API history, unavailable original IDs and historical receipts prevent a claim of complete one-command cold reproduction. No punitive, employment, educational-integrity or legal conclusion should rely solely on these probabilities.

## Frozen numerical provenance

| Record | SHA256 |
|---|---|
| V2 protocol frozen before adaptation training/scores | `276c4757774c0f1867470a040c0a303209a6a2444e8481d4402e4693ce221da4` |
| Binary blend | `df0a837b1ce7016d56373d06f822fa48d36a9e969bd525051fd1ae83ceef4628` |
| Binary calibration | `31330e39f471587ad50156c041717b326b1cbfcd6e0230ef6ff32887c653d50c` |
| Final acceptance | `51c8264e7cf57354f29d5f094d3061a4dd6f50e87238139ffb446d8c784a5215` |
| Aggregate model-card metrics extraction | `1c7afbacdceb9796472a010bc4bb64a3251cf06da3084db9b695a59d041a38e1` |

Before final metrics, an independent storage-compatibility amendment accepted only the pinned statistical component's nonnegative, exactly equal `int32` length versus `int64` scope-count representation. A validation-only proxy left raw arrays, probabilities, labels, calibration, thresholds and original producer files unchanged. Amendment SHA: `48a2e062c168dc115341796078032362a5a5f554196ab04421666ae43ed353d5`. This is disclosed engineering compatibility, not a statistical-method change.

The public packaging revision replaces documentation and records new manifest hashes; the four artifact hashes, numerical calibration and algorithms above remain the reference. A documentation-only repack has a new bundle identity. Original macOS CPU/MPS checks do not automatically establish parity on other execution platforms; use the public release's actual, revision-bound engineering reports.
