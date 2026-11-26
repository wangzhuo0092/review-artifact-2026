"""Run the experiments in §5 on the released comparison inputs."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import replace
from functools import partial
from itertools import product
import json
import multiprocessing
import os
from pathlib import Path

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")

import numpy as np
from scipy.stats import kendalltau, pearsonr, ttest_rel

from baselines import METHODS as BASELINES, fit_baseline
from inputs import DATA_ROOT, DATASETS, load_data
from references import humaine_scores, reference_scores
from repair import Config, VARIANTS, fit_repair
from studies import (PERTURBATION_RATIOS, PERTURBATION_SEEDS, PERTURBATIONS,
                     SCALE_COMPARISON_BUDGETS, SCALE_RATER_BUDGETS, SCALE_REPETITIONS,
                     perturb, scale, scale_seed)

MAIN = (*BASELINES, "repair")
ABLATIONS = {f"repair_{variant}": variant for variant in VARIANTS if variant != "full"}
FIELDS = ("study", "dataset", "method", "condition", "axis", "budget", "iteration", "seed",
          "status", "top1", "kendall", "coverage", "polluted_reliability", "unpolluted_reliability",
          "judge", "group", "agreement", "reliability", "rater_reliability", "error")


def _fit(data, method, config, device, crowd_settings=None):
    ordinary = data.subset(~data.golden)
    if not len(ordinary.y):
        raise ValueError("Sample contains no ordinary comparisons")
    if method == "repair" or method in ABLATIONS:
        return fit_repair(ordinary, replace(config, variant=ABLATIONS.get(method, "full")))
    golden = None
    if method == "crowd_bt" and np.any(data.golden):
        golden = data.annotator[data.golden], data.correct[data.golden].astype(bool)
    return fit_baseline(method, ordinary.a, ordinary.b, ordinary.y, ordinary.annotator,
                        len(data.items), n_annotators=len(data.annotators),
                        device=device, golden=golden, crowd_settings=crowd_settings)


def _metrics(scores, reference):
    covered = np.isfinite(scores) & np.isfinite(reference)
    top1 = float(np.nanargmax(scores) == np.nanargmax(reference)) if covered.any() else np.nan
    tau = kendalltau(scores, reference).statistic if covered.all() else np.nan
    return {"top1": top1, "kendall": tau, "coverage": float(covered.mean())}


def _mean(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if len(finite) else np.nan


def _trial(job, *, data, study, method, config, device, reference, crowd_settings):
    condition, axis, budget, iteration, seed = job
    result = dict(study=study, dataset=data.name, method=method, condition=condition,
                  axis=axis, budget=budget, iteration=iteration, seed=seed)
    try:
        polluted = np.empty(0, dtype=int)
        if study == "perturbation":
            sampled, polluted = perturb(data, condition, budget, seed, axis)
        elif study == "scaling":
            sampled = scale(data, axis, budget, seed)
        else:
            sampled = scale(data, "n_raters", len(data.annotators), seed)
        fit = _fit(sampled, method, config, device, crowd_settings)
        if study == "perturbation" and method == "crowd_bt" and fit.crowd_degenerate:
            raise RuntimeError("Degenerate Crowd-BT fit")
        result["status"] = "nonconverged" if fit.converged is False else "ok"
        result.update(_metrics(fit.scores, reference))
        if study == "perturbation" and method == "repair" and axis == "rater":
            is_polluted = np.isin(np.arange(len(data.annotators)), polluted)
            result["polluted_reliability"] = _mean(fit.reliability[is_polluted])
            result["unpolluted_reliability"] = _mean(fit.reliability[~is_polluted])
            if data.name == "ihq_all":
                observed = np.isfinite(fit.reliability)
                result["rater_reliability"] = json.dumps({
                    "polluted": fit.reliability[observed & is_polluted].tolist(),
                    "unpolluted": fit.reliability[observed & ~is_polluted].tolist(),
                }, separators=(",", ":"))
    except (ValueError, RuntimeError, OSError) as exc:
        result.update(status="failed", error=str(exc))
    return result


def _workers_init(trial):
    global _worker_trial
    _worker_trial = trial


def _worker_run(job):
    return _worker_trial(job)


def _condition_trials(args, data, method, config, jobs, reference, crowd_settings=None):
    trial = partial(_trial, data=data, study=args.study, method=method, config=config,
                    device=args.device, reference=reference, crowd_settings=crowd_settings)
    if args.workers == 1:
        yield from map(trial, jobs)
        return
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_workers_init,
                             initargs=(trial,), mp_context=multiprocessing.get_context("spawn")) as pool:
        yield from pool.map(_worker_run, jobs, chunksize=4)


def _stability_reference(args, data):
    if data.name == "humaine":
        return (reference_scores(args.humaine_reference, data.items)
                if args.humaine_reference else humaine_scores(data.items))
    if data.name.startswith("ihq"):
        return reference_scores(args.clic_reference, data.items)
    return None


def _clean_reference(args, data, method, config, crowd_settings=None):
    fit = _fit(data, method, config, args.device, crowd_settings)
    if fit.converged is False:
        raise RuntimeError(f"{data.name}/{method} clean-data reference fit did not converge")
    if not np.isfinite(fit.scores).all():
        raise RuntimeError(f"{data.name}/{method} clean reference does not cover all items")
    return fit.scores


def _consensus(args, data, config):
    screened = set(load_data("ihq_screened", args.data_dir).annotators)
    unscreened = set(load_data("ihq_unscreened", args.data_dir).annotators)
    if screened & unscreened or screened | unscreened != set(data.annotators):
        raise ValueError("IHQ annotator groups do not partition IHQ-all")
    fit = _fit(data, "repair", config, args.device)
    if not fit.converged:
        raise RuntimeError("The joint IHQ-all fit did not converge")
    a, b = fit.scores[data.a], fit.scores[data.b]
    agreement = np.where((a == b) | (data.y == 0.5), 0.5, (a > b) == (data.y == 1))
    for index, judge in enumerate(data.annotators):
        yield dict(study="consensus", dataset=data.name, method="repair", judge=judge,
                   group="screened" if judge in screened else "unscreened", status="ok",
                   agreement=_mean(agreement[data.annotator == index]), reliability=fit.reliability[index])


def _prior_settings(config):
    for shape, rate in product((5, 20, 100), (0.02, 0.1, 0.5, 2)):
        yield f"shape={shape},rate={rate}", "gamma", replace(config, strength_shape=shape, strength_rate=rate)
    for alpha, beta in product((2, 10, 50, 200), (2, 10, 50)):
        yield f"alpha={alpha},beta={beta}", "beta", replace(config, mean_alpha=alpha, mean_beta=beta)


def _perturbation_reference(args, data, method, config):
    if method != "crowd_bt":
        return _clean_reference(args, data, method, config), None
    # Source protocol selects the first non-degenerate clean fit in this grid.
    for alpha in (1e-5, 1e-3, 1e-2):
        settings = dict(alpha=alpha, rater_prior_strength=100)
        fit = _fit(data, method, config, args.device, settings)
        if np.isfinite(fit.scores).all() and not fit.crowd_degenerate:
            return fit.scores, settings
    raise RuntimeError("All Crowd-BT perturbation reference fits were degenerate")


def _runs(args, config, external):
    if args.study == "consensus":
        yield from _consensus(args, load_data("ihq_all", args.data_dir), config)
        return
    for name in args.datasets:
        data = load_data(name, args.data_dir)
        if args.study == "perturbation":
            data = data.subset((data.a != data.b) | data.golden)
        for method in args.methods:
            if args.study == "prior":
                for condition, axis, setting in _prior_settings(config):
                    reference = _clean_reference(args, data, method, setting)
                    count = args.repetitions or 3000
                    jobs = [(condition, axis, "", i, 20260809 + i) for i in range(count)]
                    yield from _condition_trials(args, data, method, setting, jobs, reference)
                continue
            crowd_settings = None
            try:
                if args.study == "perturbation":
                    if method == "am_elo" and (name != "ihq_all" or "rater" not in args.units):
                        continue  # am-ELO is only plotted in the IHQ rater study.
                    reference, crowd_settings = _perturbation_reference(args, data, method, config)
                else:
                    reference = external.get(name)
                    if reference is None:
                        reference = _clean_reference(args, data, method, config)
            except (RuntimeError, ValueError, OSError) as exc:
                yield dict(study=args.study, dataset=name, method=method, condition="",
                           axis="", budget="", status="reference_failed", error=str(exc))
                continue
            if args.study == "perturbation":
                seeds = PERTURBATION_SEEDS[:args.repetitions]
                units = ("rater",) if method == "am_elo" else args.units
                jobs = [(kind, unit, ratio, i, seed)
                        for kind, unit, ratio in product(PERTURBATIONS, units, PERTURBATION_RATIOS)
                        for i, seed in enumerate(seeds)]
            elif args.study == "scaling":
                jobs = [("", axis, budget, i, scale_seed(name, axis, seed_budget, i))
                        for axis, budgets in (("n_raters", SCALE_RATER_BUDGETS),
                                              ("comparisons_per_rater", SCALE_COMPARISON_BUDGETS))
                        for budget in budgets
                        for seed_budget in ("max" if axis == "n_raters" and budget == 112 else budget,)
                        for i in range(args.repetitions or SCALE_REPETITIONS)]
            else:
                count = args.repetitions or (
                    1000 if name == "humaine" or method == "crowd_bt" else 10000
                )
                jobs = [("", "", "", i, 42 + i) for i in range(count)]
            yield from _condition_trials(args, data, method, config, jobs, reference, crowd_settings)


def _summarize(records, study):
    if study == "consensus":
        seeds = np.random.SeedSequence(20260810).spawn(5)[2:4]
        output = []
        for group, seed in zip(("screened", "unscreened"), seeds):
            rng = np.random.default_rng(seed)
            pair = [(r["agreement"], r["reliability"]) for r in records if r["group"] == group]
            x, y = np.asarray(pair).T
            indices = rng.integers(0, len(x), size=(10000, len(x)))
            bootstraps = pearsonr(x[indices], y[indices], axis=1).statistic
            output.append(dict(group=group, annotators=len(x), mean_agreement=_mean(x),
                               pearson=pearsonr(x, y).statistic,
                               ci95=np.nanquantile(bootstraps, [0.025, 0.975]).tolist()))
        return output
    groups = {}
    for record in records:
        key = tuple(record[k] for k in ("dataset", "method", "condition", "axis", "budget"))
        groups.setdefault(key, []).append(record)
    summaries = []
    for key, trials in groups.items():
        valid = [trial for trial in trials if trial["status"] == "ok"]
        # Nonconverged am-ELO fits never enter the pooled cross-dataset averages.
        comparable = len(valid) == len(trials) or key[1] != "am_elo"
        row = dict(zip(("dataset", "method", "condition", "axis", "budget"), key))
        row.update(attempted=len(trials), successful=len(valid), comparable=comparable)
        for metric in ("top1", "kendall", "polluted_reliability", "unpolluted_reliability"):
            row[metric] = _mean([trial.get(metric, np.nan) for trial in valid])
        summaries.append(row)
    if study == "stability":
        for row in summaries:
            if row["method"] != "repair":
                continue
            dataset = row["dataset"]
            repairs = groups[(dataset, "repair", "", "", "")]
            for metric in ("top1", "kendall"):
                candidates = [s for s in summaries
                              if s["dataset"] == dataset and s["method"] in BASELINES
                              and s["comparable"] and np.isfinite(s[metric])]
                if not candidates:
                    continue
                best = max(candidates, key=lambda s: s[metric])["method"]
                baseline = {trial["seed"]: trial.get(metric, np.nan)
                            for trial in groups[(dataset, best, "", "", "")]
                            if trial["status"] == "ok"}
                pairs = [(trial.get(metric, np.nan), baseline.get(trial["seed"], np.nan))
                         for trial in repairs if trial["status"] == "ok"]
                complete = np.asarray([(a, b) for a, b in pairs if np.isfinite(a) and np.isfinite(b)])
                pvalue = (ttest_rel(complete[:, 0], complete[:, 1]).pvalue
                          if len(complete) > 1 and np.any(complete[:, 0] != complete[:, 1]) else np.nan)
                row[f"{metric}_comparison"] = dict(baseline=best, paired=len(complete), p=pvalue)
    return summaries


def _json_clean(value):
    if isinstance(value, list):
        return [_json_clean(entry) for entry in value]
    if isinstance(value, dict):
        return {key: _json_clean(entry) for key, entry in value.items()}
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", choices=("stability", "ablation", "perturbation", "scaling", "consensus", "prior"))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS)
    parser.add_argument("--methods", nargs="+", choices=(*MAIN, *ABLATIONS))
    parser.add_argument("--data-dir", type=Path, default=DATA_ROOT)
    parser.add_argument("--output", type=Path, required=True, help="A new CSV (with JSON summary beside it)")
    parser.add_argument("--clic-reference", type=Path, help="Official IHQ reference CSV with model_name,score")
    parser.add_argument("--humaine-reference", type=Path, help="Local copy of the public HUMAINE reference")
    parser.add_argument("--units", nargs="+", choices=("comparison", "rater"), default=("comparison", "rater"))
    parser.add_argument("--device", choices=("cpu", "auto", "cuda"), default="cpu")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--repetitions", type=int, help="Override outer trials for a shorter check")
    parser.add_argument("--reference-fits", type=int, default=20, help="Override inner bootstrap fits for a shorter check")
    args = parser.parse_args()
    if min(args.workers, args.repetitions or 1, args.reference_fits) < 1 or args.repetitions == 0:
        parser.error("Workers, repetitions and reference fits must be positive")
    if args.study == "perturbation" and args.repetitions and args.repetitions > len(PERTURBATION_SEEDS):
        parser.error("The perturbation protocol has five fixed seeds")
    allowed = {
        "stability": MAIN, "ablation": ("repair", *ABLATIONS), "perturbation": MAIN,
        "scaling": tuple(m for m in MAIN if m != "am_elo"), "consensus": ("repair",), "prior": ("repair",),
    }[args.study]
    args.methods = tuple(dict.fromkeys(args.methods or allowed))
    if any(method not in allowed for method in args.methods):
        parser.error(f"Available methods for {args.study}: {', '.join(allowed)}")
    fixed_ihq = args.study in ("scaling", "consensus", "prior")
    default_datasets = ("ihq_all",) if fixed_ihq else DATASETS
    if args.study == "perturbation":
        default_datasets = DATASETS[:6]
    args.datasets = tuple(dict.fromkeys(args.datasets or default_datasets))
    if args.study == "perturbation" and any(dataset not in DATASETS[:6] for dataset in args.datasets):
        parser.error("The perturbation study uses the six datasets, with the full IHQ set")
    if fixed_ihq and args.datasets != ("ihq_all",):
        parser.error(f"{args.study} uses IHQ-all")
    if args.study in ("stability", "ablation") and any(n.startswith("ihq") for n in args.datasets) and not args.clic_reference:
        parser.error("The IHQ stability study requires --clic-reference covering all 28 items. See README.")
    summary_path = args.output.with_suffix(".summary.json")
    if args.output.exists() or summary_path.exists():
        parser.error("Choose a new output path")
    external = {}
    if args.study in ("stability", "ablation"):
        for name in args.datasets:
            if name == "humaine" or name.startswith("ihq"):
                external[name] = _stability_reference(args, load_data(name, args.data_dir))
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    config = Config(bootstraps=args.reference_fits)
    records = []
    with output.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, FIELDS)
        writer.writeheader()
        for row in _runs(args, config, external):
            writer.writerow(row)
            records.append(row)
    summary_path.write_text(json.dumps(_json_clean(_summarize(records, args.study)), indent=2,
                                       allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {output} and {summary_path}")


if __name__ == "__main__":
    main()
