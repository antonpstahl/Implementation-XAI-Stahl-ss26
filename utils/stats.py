"""Phase-1 Inferenzstatistik: Bootstrap-CI, Cliff's delta, Wilcoxon pairwise."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def bootstrap_ci(data, stat_fn=np.mean, n_boot=2000, alpha=0.05, seed=42):
    """Bootstrap confidence interval for any statistic.

    Parameters
    ----------
    data    : array-like (NaN values are dropped)
    stat_fn : callable, default np.mean
    n_boot  : int, number of resamples
    alpha   : float, significance level (0.05 → 95 % CI)
    seed    : int, for reproducibility

    Returns
    -------
    (ci_lower, ci_upper, observed_stat)
    Parametrized for arbitrary n — reusable in Phase 3b.
    """
    rng  = np.random.default_rng(seed)
    arr  = np.asarray(data, dtype=float)
    arr  = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return np.nan, np.nan, np.nan
    observed = stat_fn(arr)
    boots = np.array([
        stat_fn(rng.choice(arr, size=len(arr), replace=True))
        for _ in range(n_boot)
    ])
    ci_lo = float(np.percentile(boots, 100 * alpha / 2))
    ci_hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return ci_lo, ci_hi, float(observed)


def cliffs_delta(x, y):
    """Cliff's delta effect size (dominance statistic) in [-1, +1].

    Thresholds (Romano et al.):
        |d| < 0.147 → negligible, < 0.33 → small, < 0.474 → medium, else large.
    Positive delta means x tends to be larger than y.
    Parametrized for arbitrary n — reusable in Phase 3b.
    """
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    y = np.asarray(y, dtype=float); y = y[~np.isnan(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan
    dominance = sum(int(xi > yi) - int(xi < yi) for xi in x for yi in y)
    return dominance / (len(x) * len(y))


def _delta_magnitude(d):
    ad = abs(d)
    if np.isnan(ad):   return 'n/a'
    if ad < 0.147:     return 'negligible'
    if ad < 0.330:     return 'small'
    if ad < 0.474:     return 'medium'
    return 'large'


def adjust_pvalues(pvalues, method='holm'):
    """Multiplizitätskorrektur für eine Familie von p-Werten (Holm oder BH).

    Bei vielen paarweisen Tests (z. B. C(k,2) Pipeline-Vergleiche) inflationiert die
    Familienfehlerrate — Holm kontrolliert die FWER, Benjamini-Hochberg die FDR.

    Parameters
    ----------
    pvalues : array-like; NaN (degenerierte Tests, n < 3) werden durchgereicht und
              zählen **nicht** zur Familiengröße.
    method  : 'holm' (FWER, Default) oder 'fdr_bh' / 'bh' (Benjamini-Hochberg, FDR).

    Returns
    -------
    np.ndarray adjustierter p-Werte in Eingabereihenfolge (NaN bleibt NaN).
    Auf [0, 1] geklippt und monoton erzwungen — verifiziert gegen statsmodels.
    """
    p = np.asarray(pvalues, dtype=float)
    out = np.full(p.shape, np.nan)
    valid = ~np.isnan(p)
    pv = p[valid]
    m = len(pv)
    if m == 0:
        return out

    order = np.argsort(pv)
    ranked = pv[order]
    if method == 'holm':
        adj = np.maximum.accumulate((m - np.arange(m)) * ranked)
    elif method in ('fdr_bh', 'bh'):
        adj = np.minimum.accumulate(((m / (np.arange(m) + 1)) * ranked)[::-1])[::-1]
    else:
        raise ValueError(f"Unbekannte Methode {method!r} (erwartet 'holm' oder 'fdr_bh').")

    res = np.empty(m)
    res[order] = np.clip(adj, 0, 1)
    out[valid] = res
    return out


def wilcoxon_pairwise(df, pipelines, metric,
                      group_col='pipeline_label',
                      id_cols=('instance_id', 'xai_model'),
                      correction='holm', alpha=0.05):
    """Pairwise Wilcoxon signed-rank tests between pipelines on paired observations.

    Parameters
    ----------
    df        : DataFrame containing group_col, metric, and id_cols
    pipelines : list of pipeline labels to compare
    metric    : str, name of the score column
    group_col : column that identifies the pipeline
    id_cols   : columns that identify a matched pair (instance × xai_model)
    correction: multiplicity correction over the pairwise family in this call
                ('holm' default, 'fdr_bh', or None to skip); see adjust_pvalues.
    alpha     : significance level for the `reject` flag on adjusted p-values.

    Returns
    -------
    DataFrame[pipeline_a, pipeline_b, n_pairs, mean_a, mean_b,
              delta_mean, statistic, p_value, cliffs_d, magnitude]
    plus, unless correction is None, [p_value_adj, reject]. Die Korrektur wird über
    die C(k,2) paarweisen Vergleiche **dieses Aufrufs** (eine Metrik) gerechnet.
    Parametrized for arbitrary n — reusable in Phase 3b.
    """
    rows = []
    for i, pa in enumerate(pipelines):
        for pb in pipelines[i + 1:]:
            da = df[df[group_col] == pa].set_index(list(id_cols))[metric]
            db = df[df[group_col] == pb].set_index(list(id_cols))[metric]
            common = da.index.intersection(db.index)
            xa, xb = da.loc[common].values, db.loc[common].values
            mask = ~(np.isnan(xa) | np.isnan(xb))
            xa, xb = xa[mask], xb[mask]
            n = int(len(xa))
            if n < 3:
                rows.append(dict(pipeline_a=pa, pipeline_b=pb, n_pairs=n,
                                 mean_a=np.nan, mean_b=np.nan, delta_mean=np.nan,
                                 statistic=np.nan, p_value=np.nan,
                                 cliffs_d=np.nan, magnitude='n/a'))
                continue
            try:
                stat, pval = scipy_stats.wilcoxon(xa, xb, alternative='two-sided')
            except ValueError:
                stat, pval = np.nan, np.nan
            cd   = cliffs_delta(xa, xb)
            rows.append(dict(
                pipeline_a=pa, pipeline_b=pb, n_pairs=n,
                mean_a=round(float(np.mean(xa)), 3),
                mean_b=round(float(np.mean(xb)), 3),
                delta_mean=round(float(np.mean(xa) - np.mean(xb)), 3),
                statistic=round(float(stat), 2) if not np.isnan(stat) else np.nan,
                p_value=round(float(pval), 4) if not np.isnan(pval) else np.nan,
                cliffs_d=round(cd, 3),
                magnitude=_delta_magnitude(cd),
            ))
    out = pd.DataFrame(rows)
    if correction and not out.empty:
        out['p_value_adj'] = adjust_pvalues(out['p_value'].values, method=correction).round(4)
        out['reject'] = out['p_value_adj'] < alpha
    return out
