# Pairwise preference inputs

The CSV files contain the ordinary pairwise judgments used in Table 1. The
three IHQ files are splits of one of the six datasets. All files are UTF-8 and
use the same four columns.

| Column | Meaning |
| --- | --- |
| `model_a` | Dataset label for the first item shown |
| `model_b` | Dataset label for the second item shown |
| `winner` | `model_a` for an A win, `model_b` for a B win, or `tie` |
| `judge` | Dataset-local annotator pseudonym, shared across the three IHQ splits |

| File | Dataset | Models | Annotators | Comparisons | Ties |
| --- | --- | ---: | ---: | ---: | ---: |
| `search_arena.csv` | Search Arena | 11 | 3,642 | 7,000 | 2,254 |
| `vision_arena.csv` | VisionArena | 17 | 13,655 | 29,849 | 11,121 |
| `chatbot_arena.csv` | Chatbot Arena | 20 | 13,383 | 33,000 | 9,706 |
| `conha.csv` | ConHa | 8 | 40 | 1,531 | 0 |
| `humaine.csv` | HUMAINE | 27 | 1,977 | 105,220 | 33,656 |
| `ihq_all.csv` | IHQ-all | 28 | 107 | 3,826 | 0 |
| `ihq_screened.csv` | IHQ-screened | 28 | 47 | 1,888 | 0 |
| `ihq_unscreened.csv` | IHQ-unscreened | 28 | 60 | 1,938 | 0 |

The files preserve the input row order, the displayed A/B orientation, model
names, and repeat comparisons. The conversion normalizes both Arena tie
variants and the HUMAINE tie marker to `tie`. Search Arena includes
comparisons where the same named model appears on both sides, as in the input
snapshot counted in Table 1. IHQ ordinary files exclude training questions.
The screened and unscreened groups have disjoint annotators and together form
IHQ-all. The files contain only the data fields used to fit rankings or group
comparisons. Annotator pseudonyms preserve their ordering in the original
fit loader. Prompts, outputs, timestamps, source row identifiers, and the
pseudonym correspondence are omitted.

The separate `ihq_calibration.csv`, `ihq_screened_calibration.csv`, and
`ihq_unscreened_calibration.csv` tables retain the original IHQ training
comparisons for Crowd-BT where the experiment uses calibration. They add two
columns. `correct` is the original binary calibration input used by Crowd-BT.
`row` is the zero-based position in the corresponding historical input after
the CSV header. To reconstruct the complete input order, place the training
comparisons at those positions and fill the remaining positions with the
ordinary comparisons in file order. The `calibration_*` aliases identify
annotators with training comparisons only. These comparisons do not enter the
Table 1 counts, RePair fits, or the selection of labels for perturbation.

Run `sha256sum --ignore-missing -c SHA256SUMS` from this directory. The
VisionArena CSV is obtained separately using the authorized reconstruction
process below. After reconstruction, `sha256sum -c SHA256SUMS` checks all inputs.

## Data sources

- Search Arena uses the 7K preference release. [Dataset card](https://huggingface.co/datasets/lmarena-ai/search-arena-v1-7k)
- VisionArena uses the battle preference release. Its access agreement includes a restriction on third-party transfer and covers annotations. Use the provider's access path for a recipient who needs to obtain the data independently. [Dataset card and access terms](https://huggingface.co/datasets/lmarena-ai/VisionArena-Battle#lmarena-visionarena-dataset-license-agreement)
- Chatbot Arena uses the 33K cleaned conversation preference release. [Dataset card](https://huggingface.co/datasets/lmsys/chatbot_arena_conversations)
- ConHa uses the preference comparisons from the conditional hallucination study. [Source paper](https://doi.org/10.1109/DCC62719.2025.00043)
- HUMAINE uses the human-feedback comparison snapshot cited in the paper. [Dataset card](https://huggingface.co/datasets/ProlificAI/humaine-evaluation-dataset)
- IHQ uses the historical 2AFC comparison snapshots distributed with the original study. Counts in current online releases can differ. [Dataset card](https://huggingface.co/datasets/Mabyduck/CLIC2024-test-human-eval), [Source paper](https://arxiv.org/abs/2510.09333)

After obtaining access to the VisionArena dataset independently, save the 31
`train-*-of-00031.parquet` shards from the provider's `data/` directory. With
`pyarrow` installed, run the following from the artifact root. The script
checks the converted file against the exact input snapshot before writing it.

```sh
python data/import_vision_arena.py /path/to/VisionArena-Battle/data
```
