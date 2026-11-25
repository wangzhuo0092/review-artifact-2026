# RePair

RePair ranks items from pairwise comparisons using estimates of annotator reliability.

## Setup

Use Python 3.11.

~~~bash
python -m pip install -r requirements.txt
~~~

Build [Crowd-BT](./_baselines/crowd_bt/README.md) if you plan to run that baseline.

## Run

~~~bash
python run.py stability --datasets conha --methods repair --output outputs/conha.csv
python run.py --help
~~~

Each run writes a CSV and a JSON summary. Use `python render.py --help` for plotting options.

See [data/README.md](./data/README.md) for input files and VisionArena setup. To run IHQ stability or ablations, pass `--clic-reference PATH` with a reference CSV covering all 28 items and containing `model_name,score` columns.
