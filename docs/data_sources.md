# Data sources and provenance

This page describes the frozen English v2 data design, its ancestral training history and its limits. Public source code and model files do not include the training corpus. Data permissions, source-code licensing and model-publication terms are separate; see [licensing](https://github.com/ydhhhh/english-ai-detector/blob/main/docs/licensing.md).

## Roles and labels

The v2 allocation was fixed before training and scores. Its 18 file views contain **84,900 unique sample IDs** in atomic roles, not 84,900 independent authors, unique parent groups or necessarily different texts. Composite views deliberately reuse records and cannot be summed as independent samples.

| Role | Human | AI | Scope |
|---|---:|---:|---|
| Training | 30,000 | 30,000 | Per class: 10,000 new + 20,000 actual-fit replay |
| New development: general | 1,000 | 1,000 | Unused MAGE components |
| New development: source pipeline | 400 | 400 | S2ORC through MAGA |
| New development: modern | 800 | 800 | Wikipedia through MAGA/MGB |
| New development: paraphrase | 300 | 600 | RAID, two rewrite origins |
| Observed legacy main development | 1,000 | 1,000 | Development-only retention |
| Observed legacy creative development | 1,000 | 1,000 | WP development-only retention |
| Combined calibration | 3,800 | 4,200 | Four fresh-source buckets + historical WP CAL replay |
| Final main | 1,500 | 1,500 | Unused MAGE components |
| Final source pipeline | 400 | 400 | Natural Questions through MAGA |
| Final modern | 1,000 | 1,000 | Wikipedia 700/S2ORC 300 paired parents |
| Final paraphrase | 600 | 1,200 | Books/news 300 parents each, two rewrite origins |

The development stress view is the union of source/modern/paraphrase development, 3,300 rows. Calibration is a union of general 2,000, S2ORC 800, Wikipedia 2,000, DIPPER 1,200 and **historical WP 2,000**. The selected historical WP records continued their CAL role; unselected relatives were not released to training or development. Calibration is not wholly fresh.

`y_ai=0` means the collector's human-source field. `y_ai=1` includes direct generation, alignment-generated text and substantive AI rewriting. In RAID, `model=human, attack=paraphrase` is an **AI positive**, not a human negative. Original author/generator, last editor and rewrite type remain different attributes. Labels are source/collector assertions, not independent certification of every author or API execution.

New MAGA/RAID family/model credibility masks are false. Custom aligned generation or an edit chain is not collapsed into an authenticated stock model version. The older 18-label catalog contains publisher/API aliases; even a label such as `gpt-6-astra` is not proof that the named provider weights were executed.

## Actual source collections

### MAGA and MGB

[MAGA](https://huggingface.co/datasets/anyangsong/MAGA/tree/34c51fc6e71561bf1203945d90d253269f9f2ab8), revision `34c51fc6e71561bf1203945d90d253269f9f2ab8`, supplied four complete MGB/MAGA train/val JSONL files. All 524,638,060 bytes passed publisher LFS size/SHA checks before parsing. There are 288,000 rows but **72,000 human-source parents**: the same human parent appears in both configurations, with one AI output per configuration. A duplicated control or a second configuration does not create another independent parent.

Publisher nominal generators include GPT-4o-mini, Gemini-2.0-flash, DeepSeek-V3, Qwen3-plus, Mistral-Medium, Hunyuan-TurboS, Llama-3.1-8B-Instruct, gemma-3-12b-it, DeepSeek-R1-0528-Qwen3-8B, Qwen3-8B, Ministral-8B-Instruct-2410 and Hunyuan-7B-Instruct. MGB also includes modern generators. MAGA's fused alignment configuration is not automatically a sentence-by-sentence DIPPER rewrite.

Only **Wikipedia, historical S2ORC and Natural Questions** were admitted from this collection, subject to the rights layers below. NQ and all connected Gemini-2.0-flash identifier parents were excluded from v2 adaptation train/development/CAL. These are source-pipeline and nominal-generator holdouts, not proof of an unseen semantic QA domain or absence from older ancestors/pretraining.

MAGA UUIDs connect records within the release but the collection process removed original upstream URLs/native IDs. UUID isolation alone therefore cannot prove independence from previous corpora. Exact text, title, prompt and near-duplicate bridges were also used.

### RAID and DIPPER

The [RAID author dataset](https://huggingface.co/datasets/liamdugan/raid/tree/865cac74188466cb0c3b7574a10204007b57a459) revision is `865cac74188466cb0c3b7574a10204007b57a459`. The acquired load used converted revision `5008b2d167f74bfd4ceff6b3b0512b7097d7cbb1`, selected column chunks in `raid/partial-train/0002`–`0009`. It is **partial official training/converted data**, not all RAID and not its hidden-label official test.

Verified HTTP ranges provided 128,196 rows: 3,561 human parents plus 124,635 target paraphrase variants, 35 variants/parent. Transferred ranges totaled 147,052,307 bytes, plus 168,219 bytes of human-title metadata. Range lengths, response status, Content-Range and SHA hashes were checked. The project did not download and independently hash the full original 11.8 GB CSV; range evidence is not whole-CSV authentication.

Human negatives require `model=human` and `attack=none`. Native `source_id`/`adv_source_id` connect originals and rewritten versions. All acquired rewrite chains matched none-attack metadata. The 1,780 news human parents matched normalized official BBC article text or text with its title removed. Books were linked by the publisher to CMU/Wikipedia plot summaries; item-by-item recovery from the CMU originals was not performed.

The final parent set has 600 clean human controls, 600 human-origin DIPPER positives and 600 AI-origin DIPPER positives. Each required view has human/AI 600/600 and shares the same controls. These views overlap their parent cohort and each other; their sizes and uncertainty intervals cannot be treated as independent or added together. This defines evidence for the registered English DIPPER setup, not all editors, translation or arbitrary human–AI mixing.

### Existing and ancestral sources

The actual-fit history was reconstructed conservatively across neural fitting, statistical fitting and vocabulary fitting: 174,019 prior fitted IDs, alongside prior development/CAL/final/pilot/observed reservations. Previously unfitted MAGE components could enter the new pool; actual-fit relatives could only be replay candidates.

The 60,000-row v2 train view has 52,800 observable connected groups. Per class, its 10,000 new records comprise MAGA 3,023, RAID 1,415 and previously unfitted MAGE 5,562. Replay has 20,000/class. WP replay contains the actual available 1,061 human records and 1,061 AI records, not invented new human examples.

Warm starts retain earlier **MAGE, HC3 English wiki/ELI5, M4 WikiHow, official Name That LLM, community Response Corpus/additions and WP** training history. Earlier acquisition also included sources used only for development/evaluation; measurement alone must not be represented as new gradient training. HC3 English medicine remained quarantined; HC3 finance was development-only, and WikiQA was external evaluation under its custom agreement.

The [community Response Corpus](https://www.kaggle.com/datasets/olegzholobov/name-that-llm-response-corpus) was an overlapping full/short compilation, not a new set of independent rows in each view. Its 31 catalog entries were Crownelius/GPT-5.6-Sol-Luna-Terra-Traces; Eventual-Inc/VLADBench-reeval; Met4physics/claude-opus-4-8-xhigh-reasoning-8.7k; ModelLineup; PawanKrd/claude-fable-5-code; Postman APIFlow-Bench raw trial transcripts; Tongyi-MAI/MobileWorld; Zek-Takai/glm53-flash-harvest; angrygiraffe/claude-opus-4.6-4.7-reasoning-8.7k; anthropic_agentic_misalignment_summer_2026; arcprize_arc_agi_v2_public_eval; artificialanalysis_microevals; artindnr/khayyam-challenge; copilot_model_value_benchmark; endless_exam; eqbench4; eqbench_creative_writing_longform; eqbench_creative_writing_v3; evaleval/EEE_datastore; glm_flash_analysis; greghavens/kimi-k3-coding-and-debugging-traces; josefchen/flavourbench_complete_core; namethatllm_public_game; novigens/token-efficiency-benchmark; pamela-zenodo-21278557; periospot_dental_benchmark; petergpt/bullshit-benchmark; rise_unibas_humanities_data_benchmark; speechmap; woog/arena-prose-100-49-models; and zlab-princeton/SWEeper-Bench-traces.

This is the **audited compilation catalog, not a claim that every entry was fitted**. Unknown-license records were not made eligible by public availability; code-only/scoped declarations and provider-output conditions were preserved. Reasoning, code/tool trajectories, refusals, truncation and cumulative conversation variants required task filtering and grouping. The separate weak-label Luna pool was not silently treated as strong nominal-version truth. Crownelius was a mirror, with Empero-derived Luna NC/attribution terms retained; its title did not establish Terra rows. The distinct [Extra API](https://www.kaggle.com/datasets/seyitkaangunes/ntl-extra-api-passages) and [Own Prompts](https://www.kaggle.com/datasets/prvsiyan/ntl-own-prompt-generations-v1) additions retained new-prompt grouping and publisher/proxy identity limits. Published call logs are not cryptographic provider authentication.

[Beemo](https://huggingface.co/datasets/toloka/beemo/tree/9c014107fe9b85c4c784c1ce3a43b0b7b0a6d162) was fully parsed for 2,187 native prompt parents. Its machine-edited outputs did not reach the preregistered 128-word-like core: GPT-4o editing max 125, Llama editing max 121. It was not admitted to the 84,900 formal records, and the length gate was not reduced to create rewrite capacity. Short/human-expert/AI-edited variants remained graph bridges; an expert-edited AI text was not automatically a human negative.

## Language, length and isolation

Structural eligibility used at least 128 Unicode regex `\b\w+\b` units, 30–3,000 whitespace units, and mechanical exclusions for visible reasoning markers, explicit truncation, suspicious endings and unclosed code fences. These rules are not complete human language/truncation adjudication and can make both exclusion and admission errors.

Fresh dev/CAL/final eligibility used independent publisher/source English evidence and structural rules, **not removal of difficult rows rejected by the deployment language guard**. Final language-rejected rows remain in class/coverage denominators. Finite-position language checks do not certify every span of a long passage.

The graph retained 645,773 old raw bridge rows, represented 399,660 old components and added new native/text nodes, resulting in 1,469,410 nodes and 474,023 components. Nodes, rows, native parents and selected samples are different counts. Excluded, short, rights-quarantined and unselected rows can remain bridges without being used in training.

Edges include native parent/source/attack/prompt identities, raw and NFKC-normalized text SHA, first-32-word fingerprints, complete/user prompts, and specific article titles. Generic/system-only prompts and ambiguous/high-frequency titles were not universal hard merge keys. A shared MAGA-Wikipedia/RAID-CMU article-title bridge was used; NQ questions were not treated as Wikipedia article titles.

Near-duplicate checks used five-word shingles over the first 3,000 lexical units, 32 CRC-MinHash values, 8×4 LSH and exact candidate Jaccard ≥0.85. The base pass checked 93,617 pairs and found 12,928 verified edges; observed-reference checks added four. The design conservatively merges static buckets ≥100, with zero such large-bucket edges in this actual build. Approximate candidate retrieval is not exhaustive semantic or hidden-prompt deduplication.

Seven normalized texts with conflicting binary labels were excluded from new roles but retained as bridges. Previously exposed Beemo examples, upstream README examples, the 36-row qualitative review and the conservative first-100 RAID reference were reserved against fresh use. Unselected siblings inherit component reservations. All ten old v1 final cohorts are permanently observed historical diagnostics and cannot become new adaptation training/CAL/development or new blind evidence.

The later audit of the **old** graph found eight multi-role components excluding pilot (three train/development, four train/final and one CAL/final), or 1,133 when pilot and full actual-fit history were included. Zero direct ID intersection had not established parent-group independence. These known conflicts were protected against new uses; old files and results were not rewritten as a new blind test. Observable isolation still cannot prove absence of hidden parents, semantic overlap or general-pretraining overlap.

WikiHow and variants were removed from all new/replay v2 data after the publisher mirror's removal notice. The warm starts preserve older exposure; this is not unlearning. Base pretraining itself is not certified free of WikiHow, evaluation topics or other overlap.

## Layered source terms

| Layer | Recorded grant and boundary |
|---|---|
| MAGA/RAID outer compilations | MIT declarations do not relicense human originals or all provider/content rights. |
| Wikipedia | Applicable historical/current CC-BY-SA text and article attribution; not complete item-level certification. [Wikimedia terms](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use#7._Licensing_of_Content). |
| Natural Questions | CC-BY-SA 3.0 data, separate from repository code. Final-only in v2 adaptation. [Official pinned README](https://github.com/google-research-datasets/natural-questions/blob/be01ce2dfdf4784010b00d1e0d28d8d789dd9c06/nq_open/README.md). |
| Historical S2ORC | Historical CC-BY-NC 4.0; current API ODC-By is not substituted. Mirror provenance does not authenticate each paper's rights. [Official historical README](https://github.com/allenai/s2orc/blob/2853285ddad5a6ebed1a65d817bfd7de1eb457c8/README.md). |
| RAID books | CMU/Wikipedia plot summaries, CC-BY-SA 3.0 US source mapping, with CMU/author/Wikipedia attribution. [CMU dataset](https://www.cs.cmu.edu/~dbamman/booksummaries.html). |
| RAID news | BBC copyright; official distribution grants research/noncommercial use, not automatic raw-text republication. [Distribution page](http://mlg.ucd.ie/datasets/bbc.html). |
| MAGE | Repository/HF Apache-2.0 versus README CC-BY-4.0 scope difference remains disclosed; original human-source terms remain relevant. [Official repository](https://github.com/yafuly/DeepfakeTextDetect). |
| HC3 wiki/ELI5 | HC3 CC-BY-SA declaration defers to stricter source terms; the ELI5 BSD repository label does not independently clear all Reddit text rights. [HC3 source table](https://github.com/Hello-SimpleAI/chatgpt-comparison-detection/tree/1f8c15c28f87e09a5abfd86ee6e15005dc7d2119#dataset-copyright). |
| M4/WikiHow ancestry | Retain recorded WikiHow NC/SA provenance and M4 attribution; v2 exclusion does not erase parent weights' history. [M4](https://github.com/mbzuai-nlp/M4/tree/628bf0fcb2e8c6b7ffa71fd1af6be413aced8f7d). |
| Official NTL/community/additions | Local official record says CC-BY-SA 4.0; current official agreement/snapshot is not fully independently authenticated. Community rows retain mixed upstream licenses/output terms; Own Prompts license markers conflict. Empero's modified MIT requires noncommercial use and visible attribution when applicable. |

Unknown rights were not converted into authorization merely because the project is noncommercial. The English medicine quarantine is distinct from a separate Chinese medicine NC dataset. Permissions for a dataset are not mechanically assigned as one license for all trained weights. The public model's Research-Use Model Notice grants only project-controlled rights and retains explicit third-party uncertainty.

## Reproduction limits and fixed identities

The actual v2 data manifest SHA is `941f568be18d361c9c3a32116ba35c0acba3d35b275e07a68c8ab4dae90665d9`; train SHA is `f7d194d69b0059e836ef0564c6fe9757624a2554fd92eb43ae537c54f3856183`. The root training/selection protocol SHA is `276c4757774c0f1867470a040c0a303209a6a2444e8481d4402e4693ce221da4`.

A fresh clone does not obtain the original corpora, official competition inputs or every historical raw API response. Upstream row APIs may not serve an immutable historical revision; cached body hashes and response revisions verify acquired history but cannot force live services to reproduce it. Source archives deliberately omit bodies/caches and use metadata-only evidence. Historical QA/manifest receipts are references, not approval seals for a new build. A cold rebuild needs authorized source access, restored inputs, independent role/graph QA and new execution bindings; complete one-command cold reproduction is not claimed. The clean public serving package is not the full research training workspace.
