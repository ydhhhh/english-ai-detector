# Training, selection and calibration

This page records the completed frozen v2 research recipe. It does not train a model when the public inference package starts. The public package consumes the complete released four-component bundle, not a single encoder checkpoint. The original research workspace, authorized corpora, parent checkpoints, immutable protocols and independently verified role allocations are required to repeat the training history; the lightweight serving package alone is not a complete cold-training distribution.

## Immutable inputs and ancestry

The v2 protocol was frozen before adaptation training and new scores:
`276c4757774c0f1867470a040c0a303209a6a2444e8481d4402e4693ce221da4`.
The data manifest is `941f568be18d361c9c3a32116ba35c0acba3d35b275e07a68c8ab4dae90665d9`.
The train view has 60,000 rows, 57,231,994 bytes, SHA256 `f7d194d69b0059e836ef0564c6fe9757624a2554fd92eb43ae537c54f3856183`.

Training is class balanced at 30,000 human/30,000 AI. Each class has 10,000 new and 20,000 replay records. There are 52,800 observable connected groups. WP replay has 1,061 human and 1,061 AI records. Native/text/title/approximate lexical grouping precedes allocation; old observed-final components and relatives cannot become new adaptation training, CAL or development. More detail and the limits of isolation are in [data sources](https://github.com/ydhhhh/english-ai-detector/blob/main/docs/data_sources.md).

| Job | Parent | Parent trained-weight SHA256 |
|---|---|---|
| Mini4 | `minilm_v3_epoch2` | `9d2ca9d251aeb2f2067478c12c564be863c478b91c579e4ea312f5170c04aabb` |
| Modern3 | `modernbert_epoch2` | `8c884a1c0172e6fe218292190f5792353fa0b4ebd27d0dba90b99928f1d4b4c8` |

Mini3 descends from Mini2 epoch 2, which warm-started the v1 baseline epoch-1 best checkpoint. Mini3 performed two epochs on 56,104 replay/WP records. Modern's parent performed two epochs on a fixed 60,000-row v1 mixed-data sample from the original public ModernBERT-base initialization. The complete chains retain earlier MAGE, HC3, M4/WikiHow, community, competition and WP exposure. The v2 60,000 records are not the models' entire training history.

The initial encoders were [MiniLM revision `1110a243...`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/1110a243fdf4706b3f48f1d95db1a4f5529b4d41) and [ModernBERT revision `8949b909...`](https://huggingface.co/answerdotai/ModernBERT-base/tree/8949b909ec900327062f0ebf497f51aef5e6f0c8), with full pretrained safetensors SHA256 respectively `53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db` and `340ac08b74eef0d7bdec2d7981a6a3d4249bf0e6aab60634b72ad02c2b8023a9`.

WikiHow was excluded from new/replay v2 input after the source mirror's removal record, but parent weights retain previous exposure. No unlearning, clean pretraining or complete pretraining-overlap exclusion is claimed.

## Binary-only adaptation

The initializer adapter copied necessary parent model/config/tokenizer files into a new immutable directory. It changed only the registered family/version loss settings to zero; class order, head shapes, windows and weights stayed fixed. It did not copy an optimizer or calibration file.

The unchanged trainer used a fresh optimizer, scheduler and RNG, with optimizer state empty and global step zero. This is warm initialization, not resuming the old schedule. Binary loss weight was 1; family/version loss weights were literally **0/0**. Credibility masks and original labels were not falsified to achieve zero fine loss. The encoder still updates from binary gradients, so new fine logits are not compatible substitutes for the old source-calibrated head.

| Setting | Mini4 | Modern3 |
|---|---:|---:|
| Epochs / rows | 2 / 60,000 | 2 / 60,000 |
| Batch / training precision | 32 / FP32 | 16 / BF16 |
| Learning rate | 5e-6 | 1e-5 |
| Weight mode | `class_domain_v3` | `class_normalized_v2` |
| Steps per epoch / total | 1,875 / 3,750 | 3,750 / 7,500 |

Shared settings: seed 20261018, AdamW weight decay 0.01, warmup ratio 0.06, gradient clip 1, six CPU threads, SDPA attention, max sequence length 512, stride 448, recovery checkpoint every 1,000 steps. The two jobs ran serially on MPS; Modern batch 16 was selected from a prior synthetic resource probe, not true-quality scores.

Both modes first apply source smoothing. Mini4 assigns total weight mass 30,000/class and 30% WP/70% other mass inside each class. Modern normalizes total smoothed mass to 30,000/class. Post-normalization weights are not clipped back to four. These are whole-table loss-weight masses, not guaranteed per-batch composition or exact realized gradient proportions.

Training randomly samples one window from each document's complete token stream. Development and inference cover the whole stream and combine window logits by newly covered token mass; long-text tails are not silently discarded. All development, CAL, final and deployed **neural forward computations use FP32**, including the Modern candidate trained in BF16. Statistical margins and probability/calibration arithmetic can use float64; FP32 does not describe every operation in the pipeline.

Each job produced immutable epoch-1 and epoch-2 snapshots. Epoch 1 copies the first epoch's best checkpoint; epoch 2 copies the final epoch-2 last checkpoint. Four candidate weight hashes are:

| Candidate | SHA256 |
|---|---|
| `minilm4_epoch1` | `e1877e1665c31a6070cfe0f0f6a5b2cea051df9382c04ebe9fb98b2dddb465e0` |
| `minilm4_epoch2` | `3393b04af27f6d82be140e541b9d7edde90660e5868a381ce759ec026d722308` |
| `modern3_epoch1` | `f85b3661178d5895498c2bb9549f07f06e033b9d7c8b2e39f3eb8642fdec33cf` |
| `modern3_epoch2` | `84bbb9e8efe59cbdc050940fa7f63bf4d5b493d27cea84fd7d6f68dbb7649f25` |

## Development-only selection

Training monitored general development (2,000 rows) and a 3,300-row stress union of source/modern/paraphrase development. Selection exported every candidate in FP32 on six cohorts: general 2,000; source pipeline 800; modern 1,600; paraphrase 900; observed legacy main 2,000; observed legacy creative 2,000. These are 24 exports, not 24 trained models. The two legacy cohorts are development retention views, not fresh final evidence.

Each cohort's connected groups were deterministically split into fit and selection halves. Only the fit half estimated margin standardization or an optional nonnegative logistic blend. The selection objective maximized the minimum cohort value of `0.5*AUROC + 0.5*recall at empirical human FPR<=2%`, then used the frozen mean/NLL/nonzero-component tie breakers. Selection did not filter rows by the runtime language guard.

The selected blend is **0.25 Mini4 epoch 1 + 0.75 Modern3 epoch 2**, intercept zero, applied to standardized raw AI margins. Other epoch-candidate weights are zero. The six selection-half core AUROCs were 0.998718, 0.999296, 0.999938, 0.994984, 0.999719 and 0.999707; corresponding per-cohort empirical 2%-FPR recalls were 0.991968, 0.990148, 1.000000, 0.990323, 0.997980 and 0.994071. These are development comparisons with local empirical cutoffs, not whole-cohort deployed or final results.

Selection freeze SHA: `b58ad4262e6551007b97936d3dd292cb2d24e14c69b7c8554f7ca8ee84cff38d`; binary blend SHA: `df0a837b1ce7016d56373d06f822fa48d36a9e969bd525051fd1ae83ceef4628`. Only after this freeze was binary calibration fitted.

## Registered mixed calibration

The combined CAL has 8,000 rows, human 3,800/AI 4,200: 6,000 fresh-source rows plus 2,000 **previously observed WP CAL replay** rows. Historical CAL remained CAL and was never released for training or selection. CAL input SHA is `7e1b5fb9f27855a0083b341708107d3720d823b5b62cc97f8d9a2ed570128f17`.

Only the supported-English core subset, 7,992 rows (human 3,795/AI 4,197), fitted BinaryPlatt and both directional cutoffs. The eight excluded rows retain their original identity; final evaluation never removes unsupported rows from error/recall denominators.

The map is `P(AI)=sigmoid(5.880817144501673*blend_margin + 0.757325435191797)`. AI requires `P(AI)>=0.7233173320180257`; human requires `1-P(AI)>=0.9049688379295407`. Both require English, core length, strict majority and the exclusive directional rule; otherwise inference abstains. CAL human→AI is 37/3,795 and AI→human is 41/4,197. These are empirical mixed-CAL budgets, not per-domain 1% or population guarantees.

Only **binary** calibration was fitted. Source hierarchy/refinement/temperatures were inherited from source CAL SHA `1b3cac20a654f1b0cb27f2784be172b9e59fa21b4858f1e5d2cd253337521ff4`. Its source-object digest remains `0d88a64119dcbb604ebacf57ea8ec15f3759ea3668f0f3655fb52004e8c67109`. Old family/model cutoffs are inactive, and both source assertion switches remain off.

Final calibration SHA is `31330e39f471587ad50156c041717b326b1cbfcd6e0230ef6ff32887c653d50c`; fit-manifest SHA is `a481882c7ef98726ad32d19cda19d00649215129c0b913a8fff6f48404d7e157`.

## Final evaluation and disclosed compatibility

The same frozen model, thresholds and language guard evaluated four final cohorts, 3,000/800/2,000/1,800 rows, plus the two overlapping 1,200-row rewrite views. Every cohort/view had to pass all registered point criteria; a high pooled average could not compensate for a failed necessary population. Each final report uses 500 whole-group bootstrap replicates, seed 42, with fixed calibration. Results and uncertainty limits are in the [model card](https://github.com/ydhhhh/english-ai-detector/blob/main/docs/model_card.md).

Before final metrics, all four evaluations stopped at an integer-storage validation mismatch: the pinned statistical exporter stores `lengths` as int32 and language scope counts as int64, with exactly equal nonnegative counts. Twelve neural raw exports matched even in storage bytes. The independent compatibility wrapper accepted only this pinned, same-shape, exact-value case using a validator-only shallow proxy. It did not change raw arrays, probabilities, labels, calibration, cutoffs, original producer code or any metric/bootstrap arithmetic. Original and compatibility code identities are both recorded; no historical SHA was rewritten.

Actual amendment SHA: `48a2e062c168dc115341796078032362a5a5f554196ab04421666ae43ed353d5`. Compatibility source SHA: `902fee6c7cb7f47015043a0120c91beef4c2e19fa10a1e11388889665e1f259a`. Acceptance SHA: `51c8264e7cf57354f29d5f094d3061a4dd6f50e87238139ffb446d8c784a5215`; status `preregistered_practical_targets_pass`.

## Research replay order and publication boundary

The original research recipe is: recover permitted input bytes and parent checkpoints; rebuild/verify the observable isolation graph and role QA; bind a new immutable protocol; prepare binary-only initializers; train Mini4 then Modern3 serially; snapshot both epochs; export all 24 development payloads; select/freeze the blend; export raw CAL; fit binary-only calibration using the two nonzero binary components; export all four required components on final roles; evaluate each suite/view with the disclosed compatibility layer; run acceptance and the predefined descriptive breakdown; package a complete inference bundle.

Historical coordinator/module names (`experiments.english_v2.train_registered`, `run_development`, `binary_finalize`, `count_storage_compat`, and the generic v1 raw exporter) identify the original research workflow. Do not assume those training tools or the original authorized datasets are all included in the lightweight public serving wheel. Existing immutable outputs must not be overwritten or their checks bypassed by editing expected hashes.

Original source receipts and dynamic API history do not guarantee one-command cold reproduction. Missing upstream native IDs, historical service revisions, separate competition acquisition, omitted corpus caches and independent-QA rebinding remain disclosed requirements. General pretraining overlap and all hidden semantic/parent relations are not certified absent. Code Apache-2.0 does not relicense corpora or determine the whole-model license.

The public packaging revision changes documentation and integrity metadata only. Numerical weight/calibration/threshold identities remain as recorded above. Repackaging and a renamed API require revision-bound engineering parity checks; old macOS CPU/MPS evidence alone does not establish Linux/CUDA/ZeroGPU equivalence.
