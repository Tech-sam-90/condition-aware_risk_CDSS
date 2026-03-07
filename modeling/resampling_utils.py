import numpy as np
import pandas as pd


def compute_resample_plan(
    n_pos: int,
    n_neg: int,
    target_minority_ratio: float = 0.15,
    oversample_multiplier: float = 2.0,
):
    if not 0.0 < target_minority_ratio < 1.0:
        raise ValueError("target_minority_ratio must be in (0, 1).")
    if oversample_multiplier < 1.0:
        raise ValueError("oversample_multiplier must be >= 1.0.")

    if n_pos <= 0 or n_neg <= 0:
        return {
            "n_pos_original": int(n_pos),
            "n_neg_original": int(n_neg),
            "n_pos_sample": int(max(n_pos, 0)),
            "n_neg_sample": int(max(n_neg, 0)),
            "target_minority_ratio": float(target_minority_ratio),
            "achieved_minority_ratio": float(n_pos / max(n_pos + n_neg, 1)),
            "oversample_multiplier": float(oversample_multiplier),
            "pos_replace": False,
            "neg_replace": False,
        }

    desired_pos = int(np.ceil(max(float(n_pos), float(n_pos) * float(oversample_multiplier))))
    desired_neg = int(np.round(desired_pos * (1.0 - target_minority_ratio) / target_minority_ratio))
    desired_neg = max(desired_neg, 1)

    # If this asks for more negatives than available, clamp negatives and re-solve positives.
    if desired_neg > n_neg:
        desired_neg = int(n_neg)
        desired_pos = int(np.round(desired_neg * target_minority_ratio / (1.0 - target_minority_ratio)))
        desired_pos = max(desired_pos, int(n_pos))

    achieved = float(desired_pos / max(desired_pos + desired_neg, 1))
    return {
        "n_pos_original": int(n_pos),
        "n_neg_original": int(n_neg),
        "n_pos_sample": int(desired_pos),
        "n_neg_sample": int(desired_neg),
        "target_minority_ratio": float(target_minority_ratio),
        "achieved_minority_ratio": achieved,
        "oversample_multiplier": float(oversample_multiplier),
        "pos_replace": bool(desired_pos > n_pos),
        "neg_replace": bool(desired_neg > n_neg),
    }


def rebalance_binary_dataframe(
    df: pd.DataFrame,
    target_col: str,
    target_minority_ratio: float = 0.15,
    oversample_multiplier: float = 2.0,
    random_state: int = 42,
):
    if target_col not in df.columns:
        raise KeyError(f"{target_col} is not a column in the training dataframe.")

    y = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int).to_numpy()
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    plan = compute_resample_plan(
        n_pos=len(pos_idx),
        n_neg=len(neg_idx),
        target_minority_ratio=target_minority_ratio,
        oversample_multiplier=oversample_multiplier,
    )

    if len(pos_idx) == 0 or len(neg_idx) == 0:
        out = df.reset_index(drop=True).copy()
        return out, plan

    rng = np.random.default_rng(random_state)

    sampled_pos = rng.choice(
        pos_idx,
        size=int(plan["n_pos_sample"]),
        replace=bool(plan["pos_replace"]),
    )
    sampled_neg = rng.choice(
        neg_idx,
        size=int(plan["n_neg_sample"]),
        replace=bool(plan["neg_replace"]),
    )

    sampled_idx = np.concatenate([sampled_pos, sampled_neg])
    rng.shuffle(sampled_idx)

    out = df.iloc[sampled_idx].reset_index(drop=True).copy()
    return out, plan


def rebalance_binary_arrays(
    X,
    y,
    target_minority_ratio: float = 0.15,
    oversample_multiplier: float = 2.0,
    random_state: int = 42,
):
    y_arr = np.asarray(y).astype(int)
    pos_idx = np.where(y_arr == 1)[0]
    neg_idx = np.where(y_arr == 0)[0]

    plan = compute_resample_plan(
        n_pos=len(pos_idx),
        n_neg=len(neg_idx),
        target_minority_ratio=target_minority_ratio,
        oversample_multiplier=oversample_multiplier,
    )

    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return X, y_arr, plan

    rng = np.random.default_rng(random_state)

    sampled_pos = rng.choice(
        pos_idx,
        size=int(plan["n_pos_sample"]),
        replace=bool(plan["pos_replace"]),
    )
    sampled_neg = rng.choice(
        neg_idx,
        size=int(plan["n_neg_sample"]),
        replace=bool(plan["neg_replace"]),
    )

    sampled_idx = np.concatenate([sampled_pos, sampled_neg])
    rng.shuffle(sampled_idx)

    X_arr = np.asarray(X)
    return X_arr[sampled_idx], y_arr[sampled_idx], plan
