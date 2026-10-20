"""Paired query statistics; seeds are never pooled as independent queries."""
from __future__ import annotations
import math
import random
from collections import defaultdict


def exact_mcnemar(improved, regressed):
    if any(type(n) is not int or n < 0 for n in (improved, regressed)):
        raise ValueError('Discordant counts must be nonnegative integers')
    n, k = improved + regressed, min(improved, regressed)
    if not n:
        return 1.0
    terms = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2)
             for i in range(k + 1)]
    peak = max(terms)
    return min(1.0, 2 * math.exp(peak) * math.fsum(math.exp(v - peak) for v in terms))


def holm(pvalues, family_size=None):
    """Missing planned tests count as p=1, so incomplete reports are not favored."""
    m = len(pvalues) if family_size is None else family_size
    if m < len(pvalues) or any(not math.isfinite(p) or not 0 <= p <= 1 for p in pvalues):
        raise ValueError('Invalid Holm family')
    adjusted, previous = [0.] * len(pvalues), 0.
    for rank, index in enumerate(sorted(range(len(pvalues)), key=pvalues.__getitem__)):
        previous = max(previous, min(1., (m - rank) * pvalues[index]))
        adjusted[index] = previous
    return adjusted


def query_ranks(result):
    rows = result['query_results']
    ranks = {row['query_id']: row['rank'] for row in rows}
    if len(ranks) != len(rows):
        raise ValueError('Duplicate query IDs')
    if any(rank is not None and (type(rank) is not int or rank < 1) for rank in ranks.values()):
        raise ValueError('Invalid retrieval rank')
    valid = [v for v in ranks.values() if v is not None]
    if not valid:
        raise ValueError('No valid queries')
    for k in (1, 5, 10):
        if not math.isclose(result['recall'][f'R@{k}'], sum(v <= k for v in valid) / len(valid), abs_tol=1e-10):
            raise ValueError('Recall disagrees with per-query ranks')
    return ranks


def paired(reference, trial, k=1, groups=None, bootstrap_samples=2000, seed=1701):
    left, right = query_ranks(reference), query_ranks(trial)
    if left.keys() != right.keys() or any((left[q] is None) != (right[q] is None) for q in left):
        raise ValueError('Changed query identities or positive validity')
    differences = {q: int(right[q] <= k) - int(left[q] <= k) for q in sorted(left) if left[q] is not None}
    better, worse = sum(v == 1 for v in differences.values()), sum(v == -1 for v in differences.values())
    result = dict(evaluated_queries=len(differences), improved_to_correct=better, regressed_to_wrong=worse,
                  delta_pp=100 * (better - worse) / len(differences), p_exact=exact_mcnemar(better, worse),
                  grouped_bootstrap=None)
    if groups is not None:
        if set(groups) != set(left) or any(not isinstance(g, str) or not g for g in groups.values()):
            raise ValueError('Group mapping must cover exactly all query IDs with explicit nonempty groups')
        buckets = defaultdict(list)
        for q, value in differences.items():
            buckets[groups[q]].append(value)
        totals = [(sum(values), len(values)) for _, values in sorted(buckets.items())]
        if len(totals) < 2:
            result['grouped_bootstrap'] = dict(status='insufficient_independent_groups', groups=len(totals))
        else:
            rng, samples = random.Random(seed), []
            for _ in range(bootstrap_samples):
                drawn = rng.choices(totals, k=len(totals))
                samples.append(100 * sum(v[0] for v in drawn) / sum(v[1] for v in drawn))
            samples.sort()
            result['grouped_bootstrap'] = dict(groups=len(totals), samples=bootstrap_samples, seed=seed,
                interval_95_pp=[samples[int(.025 * (len(samples)-1))], samples[int(.975 * (len(samples)-1))]],
                scope='paired cluster percentile interval, conditional on this trained seed; not multiplicity-adjusted')
    return result
