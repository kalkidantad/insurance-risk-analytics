"""EDA helpers: loss ratio, margins, summaries, and plotting utilities."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def add_loss_ratio_and_margin(
    df: pd.DataFrame,
    premium_col: str = "TotalPremium",
    claims_col: str = "TotalClaims",
) -> pd.DataFrame:
    """Append LossRatio and Margin columns (vectorised, safe for zero premium)."""
    out = df.copy()
    prem = pd.to_numeric(out[premium_col], errors="coerce").fillna(0)
    clm = pd.to_numeric(out[claims_col], errors="coerce").fillna(0)
    out["Margin"] = prem - clm
    out["LossRatio"] = np.where(prem > 0, clm / prem, np.nan)
    return out


def portfolio_loss_ratio(df: pd.DataFrame) -> float:
    """Aggregate loss ratio: sum(claims) / sum(premium)."""
    prem = pd.to_numeric(df["TotalPremium"], errors="coerce").fillna(0).sum()
    clm = pd.to_numeric(df["TotalClaims"], errors="coerce").fillna(0).sum()
    if prem <= 0:
        return float("nan")
    return float(clm / prem)


def loss_ratio_by_group(
    df: pd.DataFrame,
    group_cols: str | Iterable[str],
    premium_col: str = "TotalPremium",
    claims_col: str = "TotalClaims",
) -> pd.DataFrame:
    """Loss ratio computed as sum(claims)/sum(premium) within each group."""
    gcols = [group_cols] if isinstance(group_cols, str) else list(group_cols)
    sub = df[gcols + [premium_col, claims_col]].copy()
    sub[premium_col] = pd.to_numeric(sub[premium_col], errors="coerce").fillna(0)
    sub[claims_col] = pd.to_numeric(sub[claims_col], errors="coerce").fillna(0)
    agg = sub.groupby(gcols, dropna=False).agg({premium_col: "sum", claims_col: "sum"})
    agg["LossRatio"] = np.where(
        agg[premium_col] > 0,
        agg[claims_col] / agg[premium_col],
        np.nan,
    )
    agg["Margin"] = agg[premium_col] - agg[claims_col]
    return agg.reset_index()


def numeric_summary(df: pd.DataFrame, columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Descriptive statistics for numeric columns."""
    if columns is None:
        num = df.select_dtypes(include=["number", "bool"]).columns
        columns = [c for c in num if c in df.columns]
    return df[list(columns)].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T


def missing_report(df: pd.DataFrame) -> pd.Series:
    """Count and fraction of missing values per column."""
    na = df.isna().sum()
    frac = na / len(df) if len(df) else na * 0
    return pd.DataFrame({"missing": na, "fraction": frac}).sort_values("missing", ascending=False)


def correlation_numeric(df: pd.DataFrame, min_periods: int = 3) -> pd.DataFrame:
    """Pearson correlation matrix for numeric columns."""
    num = df.select_dtypes(include=[np.number])
    return num.corr(min_periods=min_periods)


def sample_df(df: pd.DataFrame, n: int, random_state: int = 42) -> pd.DataFrame:
    """Random sample of up to n rows (stable seed for reproducibility)."""
    if len(df) <= n:
        return df.copy()
    return df.sample(n=n, random_state=random_state)
