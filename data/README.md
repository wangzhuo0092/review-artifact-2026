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

The IHQ CSV files include the original training comparisons used for
Crowd-BT calibration. They have two additional columns: correct is the
binary calibration input and row is the zero-based position in the historical
input. Ordinary rows leave correct empty. Calibration rows are marked as
golden and are excluded from RePair fitting, perturbation selection, and the
Table 1 ordinary-comparison counts; Crowd-BT can use them when calibration is
enabled.
