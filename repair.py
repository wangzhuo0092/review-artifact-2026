"""RePair reference fitting, correction and aggregation (paper §4)."""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

VARIANTS = ("full", "posterior_mean", "no_bootstrap_averaging",
            "no_hierarchical_prior", "no_cross_fitting")


@dataclass(frozen=True)
class Config:
    folds: int = 5
    bootstraps: int = 20
    quadrature: int = 32
    seed: int = 42
    strength_shape: float = 5.0
    strength_rate: float = 0.1
    mean_alpha: float = 10.0
    mean_beta: float = 2.0
    concentration_shape: float = 4.0
    concentration_rate: float = 1 / 3
    reference_maxiter: int = 300
    variant: str = "full"


@dataclass(frozen=True)
class Fit:
    scores: np.ndarray
    reliability: np.ndarray
    soft_labels: np.ndarray
    converged: bool


@lru_cache(maxsize=None)
def _quadrature(count):
    nodes, weights = np.polynomial.legendre.leggauss(count)
    nodes = (nodes + 1) / 2
    return nodes, np.log(weights / 2), np.log(nodes), np.log1p(-nodes)


def _reference(data, multiplicity, targets, config):
    """Maximize Eq. (8), weighting each complete annotator log marginal."""
    import torch
    from torch.nn.functional import logsigmoid

    labels = data.y
    users = data.annotator
    selected = multiplicity[users] > 0
    decisive = np.flatnonzero(selected & (labels != 0.5))
    tie_rows = np.flatnonzero(selected & (labels == 0.5))
    decisive_users, grouped = np.unique(users[decisive], return_inverse=True)
    index = torch.as_tensor(grouped, dtype=torch.long)
    a = torch.as_tensor(data.a[decisive], dtype=torch.long)
    b = torch.as_tensor(data.b[decisive], dtype=torch.long)
    directions = torch.from_numpy(2 * labels[decisive] - 1)
    tie_a, tie_b = data.a[tie_rows], data.b[tie_rows]
    tie_weights = torch.from_numpy(multiplicity[users[tie_rows]].astype(float))
    directional_weights = torch.from_numpy(multiplicity[decisive_users].astype(float))
    nodes, log_weights, log_nodes, log_complements = map(torch.from_numpy, _quadrature(config.quadrature))
    n_items = len(targets)
    fixed = config.variant == "no_hierarchical_prior"

    def objective(parameters):
        par = torch.tensor(parameters, requires_grad=True, dtype=torch.float64)
        theta = par[:n_items]
        if fixed:
            mu, tau = 10 / 12, 12.0
        else:
            mu = 0.5 + 0.5 * torch.sigmoid(par[n_items])
            tau = torch.exp(par[n_items + 1])
        total = ((config.strength_shape - 1) * theta
                 - config.strength_rate * torch.exp(theta)).sum()
        if len(decisive):
            q = torch.sigmoid((theta[a] - theta[b]) * directions)
            likelihood = nodes[:, None] * q + (1 - nodes[:, None]) * (1 - q)
            group_log_likelihood = torch.zeros(
                (len(decisive_users), config.quadrature), dtype=torch.float64
            )
            group_log_likelihood.index_add_(0, index, likelihood.log().T)
            # The analytic Beta normalizer cancels in the normalized quadrature.
            log_beta = log_weights + (mu * tau - 1) * log_nodes + ((1 - mu) * tau - 1) * log_complements
            log_beta = log_beta - torch.logsumexp(log_beta, dim=0)
            log_marginal = torch.logsumexp(group_log_likelihood + log_beta, dim=1)
            total = total + torch.dot(directional_weights, log_marginal)
        if len(tie_rows):
            logits = theta[tie_a] - theta[tie_b]
            total = total + 0.5 * torch.dot(tie_weights, logsigmoid(logits) + logsigmoid(-logits))
        if not fixed:
            total = total + (config.mean_alpha - 1) * mu.log() + (config.mean_beta - 1) * torch.log1p(-mu)
            total = total + (config.concentration_shape - 1) * tau.log() - config.concentration_rate * tau
        (-total).backward()
        return -float(total.detach()), par.grad.numpy()

    initial = np.full(n_items, np.log((config.strength_shape - 1) / config.strength_rate))
    bounds = [(-20.0, 20.0)] * n_items
    if not fixed:
        population_guess = config.mean_alpha / (config.mean_alpha + config.mean_beta)
        unit_mean = np.clip(2 * population_guess - 1, 1e-10, 1 - 1e-10)
        initial = np.r_[initial, np.log(unit_mean / (1 - unit_mean)),
                        np.log(config.mean_alpha + config.mean_beta)]
        bounds += [(-12.0, 12.0), (np.log(0.05), np.log(500.0))]
        initial = np.clip(initial, np.asarray(bounds)[:, 0], np.asarray(bounds)[:, 1])
    result = minimize(objective, initial, method="L-BFGS-B", jac=True, bounds=bounds,
                      options={"maxiter": config.reference_maxiter, "ftol": 1e-10,
                               "gtol": 1e-6, "maxls": 40})
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"Reference optimization failed: {result.message}")
    mu, tau = (10 / 12, 12.0) if fixed else (0.5 + 0.5 * expit(result.x[n_items]), np.exp(result.x[n_items + 1]))
    return result.x[:n_items], mu, tau


def _correct(q, annotator, mu, tau, config):
    """Evaluate the normalized posterior from Eq. (9) and Eq. (11)."""
    nodes, log_weights, log_nodes, log_complements = _quadrature(config.quadrature)
    likelihood = nodes[:, None] * q + (1 - nodes[:, None]) * (1 - q)
    log_likelihood = np.log(likelihood)
    _, position = np.unique(annotator, return_inverse=True)
    totals = np.zeros((len(nodes), int(position.max()) + 1))
    for index, row in enumerate(log_likelihood):
        totals[index] = np.bincount(position, weights=row, minlength=totals.shape[1])
    log_beta = log_weights + (mu * tau - 1) * log_nodes + ((1 - mu) * tau - 1) * log_complements
    log_posterior = log_beta[:, None] + totals[:, position] - log_likelihood
    posterior = np.exp(log_posterior - logsumexp(log_posterior, axis=0))
    mean = nodes @ posterior
    if config.variant == "posterior_mean":
        retain = mean * q / (mean * q + (1 - mean) * (1 - q))
    else:
        retain = np.sum(posterior * nodes[:, None] * q / likelihood, axis=0)
    return retain, mean


def _aggregate(data, labels, config):
    """Fit the fixed fractional-label BT objective in Eq. (13)."""
    size = len(data.items)
    wins = np.zeros((size, size))
    np.add.at(wins, (data.a, data.b), labels)
    np.add.at(wins, (data.b, data.a), 1 - labels)
    shape = config.strength_shape - 1 + wins.sum(axis=1)
    pair_totals = wins + wins.T
    skill = np.full(size, (config.strength_shape - 1) / config.strength_rate)
    for _ in range(1000):
        rate = config.strength_rate + (pair_totals / (skill[:, None] + skill)).sum(axis=1)
        updated = shape / rate
        if np.max(np.abs(400 * np.log10(updated / skill))) < 1:
            return np.log(updated), True
        skill = updated
    return np.log(skill), False


def fit_repair(data, config=Config()):
    """Fit the full model or one of the four variants from Table 3."""
    if (config.variant not in VARIANTS or config.folds < 2 or config.bootstraps < 1
            or config.quadrature < 8 or config.strength_shape <= 1 or config.strength_rate <= 0):
        raise ValueError("Invalid RePair configuration")
    if min(config.mean_alpha, config.mean_beta, config.concentration_shape,
           config.concentration_rate) <= 0:
        raise ValueError("Prior shape and rate parameters must be positive")
    users = np.unique(data.annotator)
    if len(users) < 2:
        raise ValueError("Cross-fitting requires at least two observed annotators")
    shuffled = users.copy()
    np.random.default_rng(config.seed).shuffle(shuffled)
    folds = [shuffled[i::min(config.folds, len(users))] for i in range(min(config.folds, len(users)))]
    n_users = len(data.annotators)
    active_items = np.unique(np.r_[data.a, data.b])
    soft = np.full(len(data.y), 0.5)
    posterior_mean = np.zeros(len(data.y))
    decisive = data.y != 0.5
    repetitions = 1 if config.variant == "no_bootstrap_averaging" else config.bootstraps

    for index, held_out in enumerate(folds):
        if config.variant == "no_cross_fitting":
            other = np.concatenate(folds[index + 1:] + folds[:index])
            excluded = other[:len(held_out)]
        else:
            excluded = held_out
        reference = users[~np.isin(users, excluded)]
        rows = np.flatnonzero(np.isin(data.annotator, held_out) & decisive)
        if not len(rows):
            continue
        rng = np.random.default_rng(config.seed + 104729 * (index + 1))
        retained = np.zeros(len(rows))
        for _ in range(repetitions):
            sample = rng.choice(reference, size=len(reference), replace=True)
            counts = np.bincount(sample, minlength=n_users)
            # Include all items observed in the outer sample, even if this draw misses some.
            theta, mu, tau = _reference(data, counts, data.items, config)
            a, b = data.a[rows], data.b[rows]
            q = expit((2 * data.y[rows] - 1) * (theta[a] - theta[b]))
            keep, mean = _correct(q, data.annotator[rows], mu, tau, config)
            retained += keep
            posterior_mean[rows] += mean / repetitions
        retained /= repetitions
        soft[rows] = np.where(data.y[rows] == 1, retained, 1 - retained)

    count = np.bincount(data.annotator[decisive], minlength=n_users)
    total = np.bincount(data.annotator[decisive], weights=posterior_mean[decisive], minlength=n_users)
    reliability = np.divide(total, count, out=np.full(n_users, np.nan), where=count > 0)
    scores, converged = _aggregate(data, soft, config)
    scores[~np.isin(np.arange(len(scores)), active_items)] = np.nan
    return Fit(scores, reliability, soft, converged)
