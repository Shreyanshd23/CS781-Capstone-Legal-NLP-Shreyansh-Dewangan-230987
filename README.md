# CS781 Capstone: Indian Legal NLP

This repository contains source code for seven phases of the CS781 capstone. The phases cover legal text continuation, entity extraction, rhetorical roles, judgment outcomes, statute labels, summarization, and prior-case retrieval. Each phase has its own training and inference path. The reported competition metrics differ by phase and should not be compared directly.

| Phase | Task | Main approach | Reported primary score |
| --- | --- | --- | ---: |
| 1 | Legal text continuation | Decoder language models trained on the course corpus | BERTScore F1 0.8350 |
| 2 | Legal NER | Neural entity spans with document-level mention consistency | Strict macro-F1 0.6449 |
| 3 | Rhetorical roles | Sentence-context classifier and probability ensemble | Macro-F1 0.5560 |
| 4 | Judgment outcome | Long-document classifier and probability fusion | Macro-F1 0.8114 |
| 5 | Statute identification | LegalBERT with 100 sigmoid outputs | Macro-F1 0.2622 |
| 6 | Summarization | Dense causal language model with head/tail input packing | BERTScore F1 0.8135 |
| 7 | Prior-case retrieval | Lexical and semantic ranking with multiple case windows | Micro-F1@7 0.4453 |

## Layout

```text
phase1/  Language model architectures, corpus preparation, training, generation
phase2/  NER models, training, prediction, and E42 refinement
phase3/  Rhetorical-role model, training, ensemble, and E112 correction
phase4/  Judgment model, long-document preprocessing, and fusion
phase5/  LegalBERT Colab training notebook
phase6/  Summarization Colab notebooks and input-window variants
phase7/  Retrieval Colab notebooks and multiwindow variant
```

The repository contains no model weights, checkpoints, datasets, cached probabilities, or competition prediction files, as required for the code submission. The scripts and notebooks contain their training settings. Exact trained outputs require retraining or the corresponding checkpoint, which is not distributed here.

## Data and environment

The experiments use the official `Exploration-Lab/CS-781-Capstone` dataset and its prescribed Train, Dev, and Test splits. Dataset access may require a Hugging Face account and course approval. Download the data using the notebook cells or place the official files where the Python loaders expect them. For example, `phase4/data.py` expects `phase4/data/official/Phase_4_CJPE/{Train,Test}/`.

For the standalone scripts, use Python 3.10 or newer and install `requirements.txt`. CUDA is needed for GPU training. The Colab notebooks install or check their own dependencies; choose a GPU runtime before running them. Some notebooks prompt for a local `.pt` checkpoint or mount Google Drive. Supply the relevant checkpoint privately when prompted.

The Python scripts keep their original module imports. Add the repository root and `phase1` to `PYTHONPATH`; also add `phase2` when running Phase 2's standalone scripts. In PowerShell, from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH = "$PWD;$PWD\phase1;$PWD\phase2"
```

Check each script's command-line options with `--help`. The main entry points are:

```powershell
python phase1/train_modern_ddp.py --help
python phase1/generate_modern_phase1.py --help
python phase2/train_ner.py --help
python phase2/predict_ner.py --help
python -m phase3.train --help
python -m phase4.train_multichunk --help
```

Distributed Phase 1 training was designed for multiple GPUs. The dense decoder configuration is in `phase1/configs/legal_modern_168m.yaml`. Run the Phase 5–7 notebooks top to bottom in Colab. Their first cells describe dataset access, checkpoint input, and output paths.

## Phase methods and included submissions

### Phase 1: legal language modelling

`phase1/src/model/` contains Transformer components and four model families: a GPT-style decoder, a 168M dense decoder, an encoder–decoder trained with continuation and corruption objectives, and a sparse mixture-of-experts decoder. The dense decoder uses grouped-query attention, rotary positions, RMSNorm, and SwiGLU. Corpus scripts create a 32k byte-level BPE tokenizer and packed training streams. `train_modern_ddp.py` and `generate_modern_phase1.py` are the main dense-model scripts; the other training and generation scripts document the compared families. Training used official legal text from Acts, High Courts, Supreme Court, and Tribunals. Generation emits a continuation for each prompt.

### Phase 2: named-entity recognition

`ner_core.py` defines the token-level model, label set, windowing, and offset handling. `structured_ner.py` adds structured decoding. Training and prediction scripts produce character spans. The selected E42 path uses `improve_submission.py` to combine model outputs, route case-number predictions, and propagate selected exact mentions within a document. It checks span boundaries before writing the official prediction format. The reported strict macro-F1 is 0.6449.

### Phase 3: rhetorical roles

`src/model.py` defines the sentence-context classifier; `src/data.py` and `src/evaluate.py` handle the official format and local scoring. `train.py` trains and predicts sentence labels. The included ensemble scripts combine probabilities from trained runs. `quoted_cit_order.py` implements the bounded correction used after the E110 ensemble for E112. The reported E112 macro-F1 is 0.5560.

### Phase 4: judgment outcome

`data.py` loads and deduplicates official cases, keeping Test labels out of inference. `train_multichunk.py` defines a classifier over multiple document windows. The other scripts show tokenization, feature extraction, and probes used in experiments. `package_fusion.py` combines saved probability arrays and applies a decision threshold; E73 used threshold 0.345. The reported E73 macro-F1 is 0.8114. Recreating the exact fusion requires the trained source models or their probability outputs.

### Phase 5: statute identification

`colab_notebooks/P5_C2_legalbert_full.ipynb` fine-tunes LegalBERT with independent sigmoid outputs for 100 statute IDs. It selects a threshold on Dev macro-F1 and exports a submission file. Open it in Colab, choose a T4 GPU, provide access to the official dataset, and run cells in order. The reported best Phase 5 macro-F1 is 0.2622. This notebook uses a public pretrained encoder.

### Phase 6: summarization

`P6_E23_dense_full.ipynb` fine-tunes a dense causal language model to produce a summary from a judgment. It keeps source tokens from the beginning and end of the document. Other included notebooks explore longer sources, different target lengths, and quick postprocessing. The scored E23 baseline reached BERTScore F1 0.8111; the best reported Phase 6 score is 0.8135. Local records do not identify which included follow-up notebook produced that best score.

### Phase 7: prior-case retrieval

`P7_E23_student_retrieval_colab.ipynb` combines lexical passage and citation features with semantic representations. `P7_E23_multiwindow_manual.ipynb` represents the beginning, middle, and end of a case and trains a ranker with Train relevance labels. `P7_QUICK_multiwindow_upload_refine.ipynb` applies a later refinement. Predictions contain seven candidate IDs per query. The reported multiwindow Micro-F1@7 is 0.4453.

## Evaluation and reproducibility

Training and threshold selection use Train or Dev data as appropriate. Local Dev and Codabench scores refer to different examples. Because checkpoints are excluded from the code submission, this repository documents the implementation and training procedure rather than distributing trained models. Use Colab secrets or environment variables for dataset credentials; do not commit them.
