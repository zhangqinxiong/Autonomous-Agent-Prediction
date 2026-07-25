#!/usr/bin/env python3
"""Replicate the current system.md workflow against all 15 datasets.

Runs the agent's modeling pipeline (EDA → CatBoost/XGBoost/LightGBM 5-fold CV
→ Dirichlet ensemble blending) deterministically for each task and reports
Public, Private, and OOF scores.
"""

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parent.parent / "input"
EXCLUDE = set()

SKIP_CAT_COLS = {"row_id", "id"}


def _find_data_dir(name: str) -> Path:
    return DATA_DIR / name


def _load_dataset(dd: Path):
    import csv
    def _read(p: Path):
        with open(p, newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)
        return pd.DataFrame(rows[1:], columns=rows[0])
    train = _read(dd / "train.csv")
    test = _read(dd / "test.csv")
    sample = _read(dd / "sample_submission.csv") if (dd / "sample_submission.csv").exists() else None
    solution = _read(dd / "solution.csv") if (dd / "solution.csv").exists() else None
    return train, test, sample, solution


def _split_x_y(train, test):
    id_col = train.columns[0]
    target = train.columns[-1]
    X = train.drop(columns=[id_col, target])
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="ignore")
    y = train[target].values.astype(float)
    X_test = test.drop(columns=[id_col]) if id_col in test.columns else test.copy()
    for c in X_test.columns:
        X_test[c] = pd.to_numeric(X_test[c], errors="ignore")
    test_ids = test[id_col].copy()
    return X, y, X_test, test_ids, id_col, target


def _get_col_types(X):
    num = X.select_dtypes(include=[np.number]).columns.tolist()
    cat = [c for c in X.columns if c not in num and c not in SKIP_CAT_COLS]
    return num, cat


def _eda(X, y, X_test):
    n, p = X.shape
    num, cat = _get_col_types(X)
    miss = float(X.isna().mean().mean())
    pos_rate = float(y.mean())
    cat_ratio = len(cat) / max(p, 1)
    dup = int(X.duplicated().sum())
    z5 = z10 = 0
    if num:
        numeric = X[num].select_dtypes(include=[np.number])
        if len(numeric.columns):
            means, stds = numeric.mean(), numeric.std().replace(0, np.nan)
            for c in numeric.columns:
                if stds[c] > 0 and not np.isnan(stds[c]):
                    z = (numeric[c].astype(float) - means[c]).abs() / stds[c]
                    z5 += int((z > 5).sum())
                    z10 += int((z > 10).sum())
    if dup or z10:
        print(f"  [data] dup_rows={dup} outliers(z>10)={z10}", flush=True)
    return {"n": n, "p": p, "num": len(num), "cat": len(cat), "miss": miss, "pos_rate": pos_rate, "cat_ratio": cat_ratio}


def _preprocess(X, X_test, num, cat):
    """Median fill for numeric, 'MISSING' fill for categorical, winsorize outliers."""
    X, X_test = X.copy(), X_test.copy()
    if num:
        fill = X[num].median()
        if isinstance(fill, pd.Series):
            X[num] = X[num].fillna(fill)
            X_test[num] = X_test[num].fillna(fill)
        else:
            X[num] = X[num].fillna(fill)
            X_test[num] = X_test[num].fillna(fill)
        for c in num:
            m = X[c].median()
            X[c] = X[c].replace([np.inf, -np.inf], m)
            X_test[c] = X_test[c].replace([np.inf, -np.inf], m)
            lo, hi = X[c].quantile(0.01), X[c].quantile(0.99)
            X[c] = X[c].clip(lo, hi)
            X_test[c] = X_test[c].clip(lo, hi)
    if cat:
        for c in cat:
            X[c] = X[c].astype(str).fillna("MISSING")
            X_test[c] = X_test[c].astype(str).fillna("MISSING")
        X[cat] = X[cat].astype("category")
        X_test[cat] = X_test[cat].astype("category")
    return X, X_test


def _encode_cv_fold(X_tr, X_va, X_te, y_tr, cat_cols, smooth=20):
    """One target-encoded feature per categorical column."""
    X_tr = X_tr.copy()
    X_va = X_va.copy()
    X_te = X_te.copy()
    prior = float(y_tr.mean())
    for c in cat_cols:
        agg = X_tr[[c]].astype(str).copy()
        agg["y"] = y_tr
        grp = agg.groupby(c)["y"].agg(["sum", "count"])
        encoded = (grp["sum"] + smooth * prior) / (grp["count"] + smooth)
        te_c = f"{c}_te"
        vals_tr = X_tr[c].astype(str).map(encoded).fillna(prior).astype(float)
        X_tr[te_c] = vals_tr.values
        vals_va = X_va[c].astype(str).map(encoded).fillna(prior).astype(float)
        X_va[te_c] = vals_va.values
        vals_te = X_te[c].astype(str).map(encoded).fillna(prior).astype(float)
        X_te[te_c] = vals_te.values
    return X_tr, X_va, X_te


def _train_and_score(X, y, X_test, num, cat, seed=42, folds=5, gpu=False, name="?", small_n=False):
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, RepeatedStratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier

    dup_key = X.astype(str).agg("|".join, axis=1)
    n_dup_groups = int(dup_key.duplicated(keep=False).sum())
    if n_dup_groups > 0:
        groups = pd.factorize(dup_key)[0]
        skf = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
        use_groups = groups
        print(f"  [cv] StratifiedGroupKFold ({n_dup_groups} rows in duplicate groups)", flush=True)
    elif small_n:
        skf = RepeatedStratifiedKFold(n_splits=folds, n_repeats=2, random_state=seed)
        use_groups = None
        print(f"  [cv] RepeatedStratifiedKFold (2×{folds}-fold, small-n stabilizer)", flush=True)
    else:
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        use_groups = None

    if small_n:
        model_names = ["catboost", "hgb", "et", "lr_light", "lr_strong"]
        cb_p = dict(iterations=400, learning_rate=0.06, depth=4, l2_leaf_reg=20.0, random_strength=1.0)
        hgb_p = dict(max_iter=200, learning_rate=0.06, max_leaf_nodes=15, l2_regularization=3.0)
        et_p = dict(n_estimators=300, min_samples_leaf=5)
        lr_c = [(0.3, "lr_light"), (0.03, "lr_strong")]
        gpu_str = ""
    else:
        model_names = ["catboost", "xgboost", "lightgbm", "hgb", "et", "lr"]
        cb_p = dict(iterations=250, learning_rate=0.1, depth=6, l2_leaf_reg=3.0)
        hgb_p = dict(max_iter=300, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0)
        et_p = dict(n_estimators=250, min_samples_leaf=2)
        xgb_p = dict(n_estimators=350, learning_rate=0.08, max_depth=6, subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist")
        lgb_p = dict(n_estimators=350, learning_rate=0.08, num_leaves=31, subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0)
        lr_c = [(0.5, "lr")]
        gpu_str = " (GPU)" if gpu else ""

    oof = {n: np.zeros(len(y)) for n in model_names}
    te = {n: np.zeros(len(X_test)) for n in model_names}
    iters = {n: [] for n in model_names}

    te_cols = None
    split_args = (X, y, use_groups) if use_groups is not None else (X, y)

    for fold_i, (tr, va) in enumerate(skf.split(*split_args)):
        t_fold = time.time()
        X_tr, X_va = X.iloc[tr], X.iloc[va]

        X_tr_e, X_va_e, X_te_e = _encode_cv_fold(
            X_tr[num + cat], X_va[num + cat], X_test[num + cat], y[tr], cat)
        if te_cols is None:
            te_cols = [c for c in X_tr_e.columns if c.endswith("_te")]
            all_feats = num + cat + te_cols
            lr_feats = num + te_cols
            cat_idxs = list(range(len(num), len(num) + len(cat)))
            tag = "small_n" if small_n else "default"
            print(f"  [{name}] {tag}: {len(all_feats)}feats{gpu_str}"
                  f" | {','.join(model_names)}", flush=True)

        # --- CatBoost ---
        from catboost import CatBoostClassifier
        task_type = "GPU" if gpu else "CPU"
        X_cb = X_tr_e[all_feats].copy()
        X_cb_va = X_va_e[all_feats].copy()
        X_cb_te = X_te_e[all_feats].copy()
        cb = CatBoostClassifier(
            task_type=task_type, cat_features=cat_idxs,
            early_stopping_rounds=50, random_seed=seed, verbose=0, **cb_p)
        cb.fit(X_cb, y[tr], eval_set=(X_cb_va, y[va]))
        oof["catboost"][va] = cb.predict_proba(X_cb_va)[:, 1]
        te["catboost"] += cb.predict_proba(X_cb_te)[:, 1] / folds
        iters["catboost"].append(cb.tree_count_)

        if not small_n:
            # --- XGBoost ---
            from xgboost import XGBClassifier
            pos = int((y[tr] == 1).sum())
            neg = int((y[tr] == 0).sum())
            device = "cuda" if gpu else "cpu"
            X_xgb = X_tr_e[all_feats].copy()
            X_xgb_va = X_va_e[all_feats].copy()
            X_xgb_te = X_te_e[all_feats].copy()
            if cat:
                tr_cats = {c: X_xgb[c].cat.categories for c in cat}
                for c in cat:
                    X_xgb_va[c] = pd.Categorical(X_xgb_va[c].values, categories=tr_cats[c])
                    X_xgb_te[c] = pd.Categorical(X_xgb_te[c].values, categories=tr_cats[c])
            xgb = XGBClassifier(
                scale_pos_weight=neg / max(pos, 1), early_stopping_rounds=50,
                random_state=seed, verbosity=0, device=device,
                enable_categorical=True, **xgb_p)
            xgb.fit(X_xgb, y[tr], eval_set=[(X_xgb_va, y[va])], verbose=False)
            oof["xgboost"][va] = xgb.predict_proba(X_xgb_va)[:, 1]
            te["xgboost"] += xgb.predict_proba(X_xgb_te)[:, 1] / folds
            iters["xgboost"].append(xgb.best_iteration)

            # --- LightGBM ---
            from lightgbm import LGBMClassifier
            lgb_dev = "gpu" if gpu else "cpu"
            X_lgb = X_tr_e[all_feats].copy()
            X_lgb_va = X_va_e[all_feats].copy()
            X_lgb_te = X_te_e[all_feats].copy()
            if cat:
                tr_cats = {c: X_lgb[c].cat.categories for c in cat}
                for c in cat:
                    X_lgb_va[c] = pd.Categorical(X_lgb_va[c].values, categories=tr_cats[c])
                    X_lgb_te[c] = pd.Categorical(X_lgb_te[c].values, categories=tr_cats[c])
            lgb = LGBMClassifier(
                class_weight="balanced", early_stopping_rounds=50,
                random_state=seed, verbose=-1, device=lgb_dev, **lgb_p)
            lgb.fit(X_lgb, y[tr], eval_set=(X_lgb_va, y[va]))
            oof["lightgbm"][va] = lgb.predict_proba(X_lgb_va)[:, 1]
            te["lightgbm"] += lgb.predict_proba(X_lgb_te)[:, 1] / folds
            iters["lightgbm"].append(lgb.best_iteration_)

        # --- HistGradientBoosting (all feats, handles category dtype natively) ---
        X_hgb = X_tr_e[all_feats].copy()
        X_hgb_va = X_va_e[all_feats].copy()
        X_hgb_te = X_te_e[all_feats].copy()
        hgb = HistGradientBoostingClassifier(**hgb_p, early_stopping=True, random_state=seed)
        hgb.fit(X_hgb, y[tr])
        oof["hgb"][va] = hgb.predict_proba(X_hgb_va)[:, 1]
        te["hgb"] += hgb.predict_proba(X_hgb_te)[:, 1] / folds
        iters["hgb"].append(hgb.n_iter_)

        # --- ExtraTrees (num + te only) ---
        X_et = X_tr_e[num + te_cols].copy()
        X_et_va = X_va_e[num + te_cols].copy()
        X_et_te = X_te_e[num + te_cols].copy()
        et = ExtraTreesClassifier(**et_p, class_weight="balanced", n_jobs=-1, random_state=seed)
        et.fit(X_et.values, y[tr])
        oof["et"][va] = et.predict_proba(X_et_va.values)[:, 1]
        te["et"] += et.predict_proba(X_et_te.values)[:, 1] / folds
        iters["et"].append(et_p["n_estimators"])

        # --- LR (num + te, scaled) ---
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr_e[lr_feats].values)
        X_va_s = scaler.transform(X_va_e[lr_feats].values)
        X_te_s = scaler.transform(X_te_e[lr_feats].values)
        for c_val, lr_name in lr_c:
            mdl = LogisticRegression(C=c_val, max_iter=2000, solver="lbfgs",
                                     class_weight="balanced", random_state=seed)
            mdl.fit(X_tr_s, y[tr])
            oof[lr_name][va] = mdl.predict_proba(X_va_s)[:, 1]
            te[lr_name] += mdl.predict_proba(X_te_s)[:, 1] / folds

        total_folds = folds * (2 if small_n and n_dup_groups == 0 else 1)
        print(f"  fold {fold_i+1}/{total_folds} done [{time.time()-t_fold:.0f}s]", flush=True)

    results = {}
    test_preds = {}
    for n in model_names:
        auc = roc_auc_score(y, oof[n])
        mean_it = int(np.mean(iters[n])) if iters[n] else 2000
        print(f"  [{n}] OOF AUC={auc:.5f}  iters={mean_it}", flush=True)
        results[n] = {"oof": oof[n], "oof_auc": auc}
        test_preds[n] = te[n]

    return results, test_preds


def _dirichlet_blend(oofs, test_preds, y, n_samples=5000, seed=42):
    """Dirichlet random search for optimal blend weights."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.RandomState(seed)
    names = list(oofs.keys())
    best_w = None
    best_auc = -1
    weights = rng.dirichlet(np.ones(len(names)), size=n_samples)
    for w in weights:
        blended = sum(w[i] * oofs[n] for i, n in enumerate(names))
        auc = roc_auc_score(y, blended)
        if auc > best_auc:
            best_auc = auc
            best_w = w
    blended_te = sum(best_w[i] * test_preds[n] for i, n in enumerate(names))
    return best_w, best_auc, blended_te


def _row_key(X: pd.DataFrame) -> pd.Series:
    return X.astype(str).agg("|".join, axis=1)


def _rank(a: np.ndarray) -> np.ndarray:
    return pd.Series(a).rank(pct=True).to_numpy()


def apply_safe_overrides(pred, X, Xte, y, key_tr, alpha=5.0):
    pred = np.asarray(pred, dtype=float)
    key_te = _row_key(Xte)
    te_grp = pd.Series(pred).groupby(key_te.to_numpy()).transform("mean").to_numpy()
    n_te_dup = int(key_te.duplicated(keep=False).sum())
    pred = te_grp
    if n_te_dup:
        print(f"  [override] {n_te_dup} duplicated test rows → identical predictions", flush=True)

    g = pd.DataFrame({"k": key_tr.to_numpy(), "y": y}).groupby("k")["y"].agg(["sum", "count"])
    ybar = float(np.mean(y))
    g["purity"] = g["sum"] / g["count"]
    g["smoothed"] = (g["sum"] + alpha * ybar) / (g["count"] + alpha)
    strong = g[(g["count"] >= 2) & ((g["purity"] >= 0.9) | (g["purity"] <= 0.1))]
    if len(strong):
        m = key_te.map(strong["smoothed"])
        hit = m.notna().to_numpy()
        if hit.any():
            r = _rank(pred)
            pred = np.where(hit, 0.5 * r + 0.5 * m.fillna(0.0).to_numpy(), pred)
            print(f"  [override] {int(hit.sum())} exact train↔test matches nudged (α={alpha})", flush=True)
    return pred


def _score_submission(pred, solution, sample, id_col):
    """Score against solution.csv Public/Private splits."""
    from sklearn.metrics import roc_auc_score
    sub = pd.DataFrame({sample.columns[0]: sample[sample.columns[0]], sample.columns[-1]: pred})
    merged = sub.merge(solution, on=id_col, suffixes=("_pred", "_true"))
    pred_col = sample.columns[-1]
    true_col = [c for c in merged.columns if c.endswith("_true") and c.startswith(pred_col.split("_")[0])]
    if not true_col:
        true_col = [f"{pred_col}_true"]
    true_col = true_col[0]
    if true_col not in merged.columns:
        true_col = [c for c in merged.columns if c != pred_col and c != id_col and c != "Usage"][0]
    pred_col_merged = f"{pred_col}_pred" if f"{pred_col}_pred" in merged.columns else pred_col
    is_public = merged["Usage"] == "Public"
    is_private = merged["Usage"] == "Private"
    scores = {}
    for label, mask in [("Public", is_public), ("Private", is_private), ("Overall", slice(None))]:
        sub = merged[mask] if label != "Overall" else merged
        scores[label] = roc_auc_score(sub[true_col], sub[pred_col_merged]) if len(sub) > 0 else float("nan")
    return scores


def run_task(name: str, seed=42, folds=5, gpu=False):
    dd = _find_data_dir(name)
    if not dd.exists():
        return None

    train, test, sample, solution = _load_dataset(dd)
    X, y, X_test, test_ids, id_col, target = _split_x_y(train, test)
    num, cat = _get_col_types(X)
    
    # Remove constant columns
    const = [c for c in X.columns if X[c].nunique() <= 1]
    if const:
        print(f"  [data] dropping {len(const)} constant column(s): {const}", flush=True)
        X.drop(columns=const, inplace=True)
        X_test.drop(columns=[c for c in const if c in X_test.columns], inplace=True)
        num = [c for c in num if c not in const]
        cat = [c for c in cat if c not in const]

    # Dynamic folds: large datasets use 3-fold for speed
    n = len(X)
    if n > 20000:
        folds = 3
    elif n < 1500:
        folds = 5
    else:
        folds = 5

    fp = _eda(X, y, X_test)

    small_n = n < 1500
    t0 = time.time()
    X_proc, X_test_proc = _preprocess(X, X_test, num, cat)
    if gpu:
        try:
            import torch
            print(f"  GPU: {torch.cuda.get_device_name(0)}", flush=True)
        except Exception:
            print(f"  GPU: requested but torch not available, falling back to CPU", flush=True)
    else:
        print(f"  CPU mode", flush=True)
    tag = "small_n" if small_n else "default"
    print(f"  [{tag}] n={fp['n']}  (CB+XGB+LGB+LR ensemble)", flush=True)
    oof_results, test_preds = _train_and_score(X_proc, y, X_test_proc, num, cat, seed, folds, gpu, name, small_n)

    weights, ens_auc, ens_pred = _dirichlet_blend(
        {n: r["oof"] for n, r in oof_results.items()},
        test_preds, y
    )

    # Safe overrides (post-hoc, affects only test predictions)
    key_tr = _row_key(X[num + cat])
    ens_pred = apply_safe_overrides(ens_pred, X_proc[num + cat], X_test_proc[num + cat], y, key_tr)

    lb_scores = _score_submission(ens_pred, solution, sample, id_col)
    elapsed = time.time() - t0

    res = {"name": name, **fp,
           "public": lb_scores["Public"], "private": lb_scores["Private"],
           "overall": lb_scores["Overall"], "elapsed": elapsed,
           "ensemble_oof": ens_auc,
           "ensemble_weights": {n: float(w) for n, w in zip(oof_results.keys(), weights)}}
    for m in ["catboost", "xgboost", "lightgbm", "hgb", "et", "lr_light", "lr_strong", "lr"]:
        res[f"{m}_oof"] = oof_results.get(m, {}).get("oof_auc")
    return res


def main():
    ap = argparse.ArgumentParser(description="Test pipeline against all datasets")
    ap.add_argument("--datasets", nargs="*",
                    help="Specific datasets (default: all except train_14)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--gpu", action="store_true", help="Enable GPU acceleration")
    ap.add_argument("--output", default=None, help="Save results CSV to path")
    args = ap.parse_args()

    if args.datasets:
        datasets = [f"train_{int(d):02d}" if d.isdigit() else d for d in args.datasets]
    else:
        datasets = sorted(
            d.name for d in DATA_DIR.iterdir()
            if d.is_dir() and d.name.startswith("train_") and d.name not in EXCLUDE
        )

    rows = []
    for ds in datasets:
        print(f"\n{'=' * 60}")
        print(f"  {ds}")
        print(f"{'=' * 60}")
        sys.stdout.flush()
        result = run_task(ds, args.seed, args.folds, args.gpu)
        if result is None:
            print(f"  SKIP — {ds} not found")
            continue
        rows.append(result)
        w = result["ensemble_weights"]
        print(f"  n={result['n']} p={result['p']} num={result['num']} cat={result['cat']} "
              f"miss={result['miss']:.3f} pos={result['pos_rate']:.4f}")
        cb = result.get('catboost_oof')
        xgb = result.get('xgboost_oof')
        lgb = result.get('lightgbm_oof')
        hgb = result.get('hgb_oof')
        et = result.get('et_oof')
        lr03 = result.get('lr_light_oof') or result.get('lr_oof')
        lr003 = result.get('lr_strong_oof')
        print(f"  ", end="")
        for tag, v in [("CB",cb),("XGB",xgb),("LGB",lgb),("HGB",hgb),("ET",et),("LR",lr03),("LRs",lr003)]:
            if v is not None:
                print(f"{tag}={v:.4f} ", end="")
        print()
        print(f"  Ens OOF={result['ensemble_oof']:.4f}  weights: "
              + " ".join(f"{k}={v:.3f}" for k,v in sorted(w.items()) if v > 0.001))
        print(f"  Public={result['public']:.4f} Private={result['private']:.4f} "
              f"Overall={result['overall']:.4f}  [{result['elapsed']:.0f}s]")
        sys.stdout.flush()

    if not rows:
        print("No results.")
        return

    headers = ["CB", "XGB", "LGB", "HGB", "ET", "LR", "Ens", "Public", "Private"]
    cols = ["catboost_oof", "xgboost_oof", "lightgbm_oof", "hgb_oof",
            "et_oof", "lr_oof", "ensemble_oof", "public", "private"]
    fmt = "  {'Dataset':<12} {'Rows':>6} " + " ".join(f"{{{c}:>7}}" for c in ["CB", "XGB", "LGB", "LR", "ET", "HGB", "Ens", "Public", "Private"])
    sep = "  {'-'*12} {'-'*6} " + " ".join("{:->7}" for _ in headers)
    print(f"\n{'=' * 60}")
    print(f"  SUMMARY ({len(rows)} datasets)")
    print(f"{'=' * 60}")
    header_str = f"  {'Dataset':<12} {'Rows':>6}"
    for h in headers:
        header_str += f" {h:>7}"
    print(header_str)
    print(f"  {'-'*12} {'-'*6}" + "".join(" -------" for _ in headers))
    for r in rows:
        line = f"  {r['name']:<12} {r['n']:>6}"
        for c in cols:
            v = r.get(c)
            line += f" {v:>7.4f}" if v is not None else "     nan"
        print(line)
    print(f"  {'-'*12} {'-'*6}" + "".join(" -------" for _ in headers))
    line = f"  {'MEAN':<12} {'':>6}"
    for c in cols:
        vals = [r[c] for r in rows if r.get(c) is not None]
        line += f" {np.mean(vals):>7.4f}" if vals else "     nan"
    print(line)

    if args.output:
        df = pd.DataFrame(rows)
        df.to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
