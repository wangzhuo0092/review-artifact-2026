# RePair

Anonymous companion code and comparison inputs for *RePair: Reliable Ranking
from Noisy Crowdsourced Pairwise Preferences*. The package follows Sections 4
and 5 of the manuscript. It contains preference inputs, source code and
experiment settings. The runner writes new outputs to a path you specify.

`repair.py` implements the annotator-disjoint reference fits, the normalized
leave-one-comparison-out reliability posterior, and the final Bradley–Terry
fit using fixed fractional targets. The default settings are five folds,
20 annotator bootstrap fits per fold and 32 Gauss–Legendre quadrature nodes.
Item strengths use Gamma(5, 0.1), the population mean uses Beta(10, 2)
truncated to (0.5, 1), and the concentration uses Gamma(4, 1/3). Gamma
parameters are shape and rate. Ties contribute half a win and are excluded
from the directional reliability likelihood.

Use Python 3.11 and install the pinned dependencies from the directory root.

~~~sh
python -m pip install -r requirements.txt
(cd data && sha256sum --ignore-missing -c SHA256SUMS)
~~~

Build the vendored Crowd-BT executable if running that baseline. The build
uses CMake and a C++17 compiler. The Crowd-BT instructions describe how to
provide its pinned dependencies offline. [Crowd-BT build](./_baselines/crowd_bt/README.md)

~~~sh
cmake -S _baselines/crowd_bt -B _baselines/crowd_bt/build -DCMAKE_BUILD_TYPE=Release
cmake --build _baselines/crowd_bt/build --target elo_main --parallel 2
~~~

This short run checks the installation. Use the full defaults for evaluation.

~~~sh
python run.py stability --datasets conha --methods repair --repetitions 1 --reference-fits 1   --output /tmp/repair-smoke.csv
~~~

| Runner | Manuscript | Evaluation |
| --- | --- | --- |
| `stability` | §5.2, Table 2 | Paired resampling of complete annotator records, top-1 agreement, Kendall’s τ and paired tests |
| `ablation` | §5.2, Table 3 | The same protocol with the full model and four ablations |
| `perturbation` | §5.3 and §5.4, Figs. 2–3 | Random, Equal, Flip and Mixed label changes at the comparison and annotator levels |
| `scaling` | §5.4, Fig. 4 | IHQ-all annotator and comparison budgets compared with each method’s full-data ranking |
| `consensus` | §5.4, Fig. 5 | One joint IHQ-all fit, followed by within-group Pearson correlations and percentile bootstrap intervals |
| `prior` | §5.4, Table 4 | The Gamma and Beta grids in the paper, varying one prior family at a time |

For each run, use `python run.py STUDY --output /path/to/new.csv`. Use
`--workers` for parallel trials and `--help` to see all options. The runner
creates a CSV of trials and a JSON summary. Agreement is stored on a 0–1 scale.
Generate figures from completed runs with
`python render.py /path/to/perturbation.csv /path/to/scaling.csv /path/to/consensus.csv --output-dir /path/to/figures`.

The stability and ablation runs use the public HUMAINE leaderboard on HUMAINE
and each method’s full-data fit on Search Arena, VisionArena, Chatbot Arena
and ConHa. The HUMAINE snapshot is pinned by commit and SHA-256 and is read on
demand. An offline copy can be passed as `--humaine-reference`.

The manuscript specifies an official CLIC 2024 reference for all IHQ splits.
The available historical run records identify full-data fits or a screened
split proxy. The public challenge archive has one leaderboard per bitrate
and leaves out the uncompressed `reference.png` present in the IHQ item set.
To evaluate IHQ for Tables 2–3, supply the verified 28-item reference as
`--clic-reference /path/to/official-clic.csv`, with columns `model_name,score`
and higher scores indicating better ranks. Until that input is available,
select the five other datasets with `--datasets search_arena vision_arena
chatbot_arena conha humaine`. [Official CLIC 2024 leaderboards](https://archive.compression.cc/2024/leaderboard/image_0_075/test/)

The stability and ablation seed for trial *i* is 42 + *i*. HUMAINE uses
1,000 trials and the other datasets use 10,000, with Crowd-BT capped at 1,000
in the stability run. The comparison-level and annotator-level perturbations
share five paired seeds across the six datasets. Fig. 2 averages datasets
equally, excludes am-ELO, and uses IHQ-all as the IHQ entry. The source
perturbation loader excludes same-model pairs. Fig. 3 uses the annotator-level
IHQ-all conditions and omits reliability estimates for annotators without
directional observations. Fig. 4 draws from the 112 IHQ-all annotators in the
source scaling pool, including five with calibration comparisons only. Table 1
counts the 107 IHQ-all annotators with ordinary comparisons. Calibration
rows contribute only to Crowd-BT in the experiments that use them.

The input README documents sources and upstream access terms. VisionArena
annotations are available through the provider's access process. After
obtaining the authorized VisionArena-Battle Parquet files, use the supplied
importer to reconstruct the exact input CSV. See the data README for commands. [Input data and provenance](./data/README.md)
