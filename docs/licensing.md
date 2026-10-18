# Licensing and publication scope

The **original source code and English documentation** in this public repository are Apache-2.0, with the original project's MIT copyright and permission notice preserved. The **trained model** has a separate [Research-Use Model Notice](https://huggingface.co/ydh0822/english-ai-detector/blob/main/LICENSE-MODEL.md). **Training corpora** retain their original source-specific terms and are not redistributed here.

These scopes should not be collapsed into a single license badge. Apache source licensing does not authorize every corpus or trained artifact. Data NC/ShareAlike declarations do not automatically become the license of every trained model. This page discloses the evidence and limits rather than offering a blanket legal clearance.

## Source-code license and MIT history

The publication preserves the prior `Copyright (c) 2026 Name the LLM project contributors` MIT notice. MIT permits sublicensing subject to retaining its copyright/permission notice; Apache-2.0 covers the new public source distribution while the previous MIT grant is not withdrawn. Relevant upstream notices and complete license texts remain applicable. Source-code Apache permission is not restricted to research-only use by this model notice. [MIT text](https://opensource.org/license/mit), [Apache-2.0 text](https://www.apache.org/licenses/LICENSE-2.0).

The original frozen research bundle remains unchanged locally. A new public packaging revision replaces embedded documentation and records new manifest hashes while preserving the same four trained components, 75/25 binary blend, calibration and thresholds. A documentation-only repack has a new bundle identity; it must not reuse the old manifest hash as if bytes were unchanged.

## Pretrained models

| Material | Pinned upstream revision | Publisher declaration |
|---|---|---|
| [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/1110a243fdf4706b3f48f1d95db1a4f5529b4d41) | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` | Apache-2.0 |
| [ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base/tree/8949b909ec900327062f0ebf497f51aef5e6f0c8) | `8949b909ec900327062f0ebf497f51aef5e6f0c8` | Apache-2.0 for architecture, weights and training code |

Retain their license texts, applicable upstream notices, source attribution and identification of project fine-tuning/modifications. Apache section 4 permits additional/different terms for modifications or a derivative as a whole while retaining its requirements; the separate project notice does not revoke upstream Apache rights in the original base artifacts. No distilgpt2 weights are required: that model was an unselected development probe.

Runtime dependencies, including PyTorch, Transformers, Tokenizers, Safetensors, Hugging Face Hub, NumPy/SciPy/scikit-learn, PyArrow and langid, retain their own distribution licenses. Installing them does not relicense them as project code; their installed license/notice files remain authoritative.

## Trained-model release

The Model Hub repository uses `license: other`, named `research-use-model-notice`, linking to `LICENSE-MODEL.md`. The publishing owner grants only rights it controls for noncommercial research, teaching and evaluation, with attribution, notice retention and the stated conditions. This is a chosen project publication grant, **not an assertion that all data licenses legally attach to the numerical weights**.

The model is not offered as an unrestricted Apache-2.0 weight release. Its notice does not supply unknown third-party permissions, determine whether every trained artifact is a copyright adaptation, or prove complete lawful redistribution in every jurisdiction.

The full model contains three fine-tuned neural artifacts plus a statistical refinement artifact with its fitted vocabulary. Source heads retain older training histories even though their public outputs are rankings only. A public release is not a license to republish source corpora, historical prompts or raw API responses.

## Training-data layers retained in the disclosures

| Source/layer | Recorded terms and limits |
|---|---|
| MAGA/MGB and RAID compilations | Outer MIT declarations do not cover every human original, quoted work or generation-provider agreement. |
| Wikipedia / HC3 wiki / MAGA `wiki_intro` | Applicable historical/current CC-BY-SA text conditions, article attribution and exceptions. Source-level lineage is not item-level rights authentication. [Wikimedia policy](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use#7._Licensing_of_Content). |
| Natural Questions | Official data CC-BY-SA 3.0; code licensing is separate. NQ is v2 final-only, not adaptation training. [Pinned original README](https://github.com/google-research-datasets/natural-questions/blob/be01ce2dfdf4784010b00d1e0d28d8d789dd9c06/nq_open/README.md). |
| Historical S2ORC | Official historical CC-BY-NC 4.0. Current Semantic Scholar API ODC-By is not substituted for this mirror's history; individual paper rights were not all authenticated. [Historical official README](https://github.com/allenai/s2orc/blob/2853285ddad5a6ebed1a65d817bfd7de1eb457c8/README.md). |
| RAID books | CMU/Wikipedia plot summaries under linked CC-BY-SA 3.0 US, with author/CMU/Wikipedia attribution. Not public-domain Gutenberg data. [CMU](https://www.cs.cmu.edu/~dbamman/booksummaries.html). |
| RAID news | BBC owns original articles; the distributor grants research/noncommercial use. Text matching establishes provenance, not unrestricted article or model redistribution rights. [Original distribution](http://mlg.ucd.ie/datasets/bbc.html). |
| MAGE | Apache repository/HF markers versus a README CC-BY-4.0 badge are disclosed; their precise scope and underlying source rights are not silently resolved using the broadest label. [Repository](https://github.com/yafuly/DeepfakeTextDetect). |
| HC3 English wiki/ELI5 | HC3 defers to stricter source terms; an ELI5 repository BSD declaration does not independently establish every Reddit author's content rights. [Source table](https://github.com/Hello-SimpleAI/chatgpt-comparison-detection/tree/1f8c15c28f87e09a5abfd86ee6e15005dc7d2119#dataset-copyright). |
| M4/WikiHow history | Retain the recorded NC/SA provenance and M4 citation. V2 excludes all new/replay WikiHow parents, but warm-start ancestors retain exposure. The publisher mirror's removal record is not unlearning or automatic permission/prohibition for existing weights. [M4](https://github.com/mbzuai-nlp/M4/tree/628bf0fcb2e8c6b7ffa71fd1af6be413aced8f7d), [mirror history](https://huggingface.co/datasets/anyangsong/MAGA-human-source/blob/3f454fb602377be022d3be21b353c87723ba4963/README.md). |
| Competition/community/additions | The local official NTL record says CC-BY-SA 4.0; the exact remote snapshot and applicable model-publication rules were not fully independently authenticated. Mixed community sources retain their original/output terms; Own Prompts has conflicting CC-BY/CC-BY-SA markers. |
| Empero-derived community material | Original modified MIT permits noncommercial use/distribution with visible Leon Lehmann and Empero AI credit and source links when applicable; a wider mirror license cannot waive it. Credit: [Leon Lehmann / Empero AI](https://empero.org), [original dataset and license](https://huggingface.co/datasets/empero-ai/gpt-5.6-luna-sft-900x/blob/0d9a45f6a749cad293847ff72dc8b9393eb5330d/LICENSE). The 886-record audit count describes candidates, not a newly verified count fitted into each artifact. |

HC3 English medicine stayed quarantined; it was not reclassified as authorized NC data. HC3 finance/WikiQA and other evaluation-only material should not be described as new gradient-training sources. Beemo was not admitted to the formal v2 core views. See [data sources](https://github.com/ydhhhh/english-ai-detector/blob/main/docs/data_sources.md) for roles and ancestry.

## What remains uncertain

No reviewed primary source expressly prohibited publishing these discriminator weights. That is not proof of an unrestricted license. Remaining evidence gaps include official NTL publication provisions, exact actual-fit intersections for mixed/custom community terms, MAGE's scope declarations, Reddit content rights, WikiHow ancestor/removal history, the precise reach of the BBC grant, and the original agreements applicable when some provider outputs were collected. Current API terms are not substituted for historical collector agreements.

CC licenses apply when the relevant copyright rights require permission; exceptions, adaptations and memorization depend on facts and jurisdiction. Creative Commons distinguishes conservative compliance guidance from a legal analysis and does not establish that every model is a derivative of every training work. This project therefore retains NC/SA provenance without mechanically assigning a combined CC license to all model weights. [CC guidance](https://creativecommons.org/using-cc-licensed-works-for-ai-training-2/).

The Research-Use Model Notice does not settle those gaps or grant rights controlled by third parties. It records a limited project grant and the scope of the intended noncommercial release. Commercial deployment, broad relicensing or redistribution of source text requires its own applicable rights analysis and any necessary permissions. Source publication and research transparency do not constitute legal certification.
