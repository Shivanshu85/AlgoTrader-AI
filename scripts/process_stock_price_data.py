"""
Process stock OHLCV CSV data into model-ready features and targets.

Example:
python scripts/process_stock_price_data.py ^
  --input data/raw/AAPL.csv ^
  --output data/processed/AAPL_processed.csv ^
  --ticker AAPL
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ["date", "open", "high", "low", "close", "volume"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process stock price CSV into training-ready output.")
    parser.add_argument("--input", required=True, help="Path to raw input CSV file.")
    parser.add_argument(
        "--output",
        default="data/processed/stock_data_processed.csv",
        help="Path for processed output CSV.",
    )
    parser.add_argument(
        "--ticker",
        default=None,
        help="Ticker symbol override. If omitted, inferred from input filename.",
    )
    parser.add_argument(
        "--prediction-horizon",
        type=int,
        default=1,
        help="Number of days ahead for target generation.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Temporal train split ratio.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Temporal validation split ratio.",
    )
    return parser.parse_args()


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Adj_Close": "adj_close",
        "Volume": "volume",
        "Ticker": "ticker",
    }
    out = df.rename(columns=rename_map)
    unnamed_cols = [c for c in out.columns if str(c).startswith("Unnamed:")]
    if unnamed_cols and "date" not in out.columns:
        out = out.rename(columns={unnamed_cols[0]: "date"})
    return out


def _validate_input(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _reshape_close_only_wide(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle datasets shaped like:
    date, Stock_1, Stock_2, ...
    """
    if "date" not in df.columns:
        return df

    if set(REQUIRED_COLUMNS).issubset(df.columns):
        return df

    value_cols = [c for c in df.columns if c != "date"]
    if not value_cols:
        return df

    long_df = df.melt(id_vars=["date"], value_vars=value_cols, var_name="ticker", value_name="close")
    long_df["open"] = long_df["close"]
    long_df["high"] = long_df["close"]
    long_df["low"] = long_df["close"]
    long_df["volume"] = 0
    return long_df


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["returns"] = out["close"].pct_change()
    out["log_returns"] = np.log(out["close"] / out["close"].shift(1))

    out["sma_5"] = out["close"].rolling(5).mean()
    out["sma_10"] = out["close"].rolling(10).mean()
    out["sma_20"] = out["close"].rolling(20).mean()
    out["ema_12"] = out["close"].ewm(span=12, adjust=False).mean()
    out["ema_26"] = out["close"].ewm(span=26, adjust=False).mean()
    out["macd"] = out["ema_12"] - out["ema_26"]
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["rsi_14"] = _compute_rsi(out["close"], 14)

    rolling_std_20 = out["close"].rolling(20).std()
    out["bb_mid"] = out["sma_20"]
    out["bb_upper"] = out["bb_mid"] + (2 * rolling_std_20)
    out["bb_lower"] = out["bb_mid"] - (2 * rolling_std_20)

    out["price_range"] = out["high"] - out["low"]
    out["price_range_pct"] = out["price_range"] / out["close"].replace(0, np.nan)
    out["volume_change"] = out["volume"].pct_change()
    out["volatility_10"] = out["returns"].rolling(10).std()

    return out


def _create_targets(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    out = df.copy()
    out["target_close"] = out["close"].shift(-horizon)
    out["target_return"] = (out["target_close"] - out["close"]) / out["close"].replace(0, np.nan)
    out["target_direction"] = (out["target_return"] > 0).astype(int)
    return out


def _temporal_split(df: pd.DataFrame, train_ratio: float, val_ratio: float) -> Dict[str, pd.DataFrame]:
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return {
        "train": df.iloc[:train_end].copy(),
        "validation": df.iloc[train_end:val_end].copy(),
        "test": df.iloc[val_end:].copy(),
    }


def process_file(
    input_path: Path,
    output_path: Path,
    ticker: str | None,
    prediction_horizon: int,
    train_ratio: float,
    val_ratio: float,
) -> Dict[str, str]:
    raw_df = pd.read_csv(input_path)
    raw_df = _normalize_columns(raw_df)
    raw_df = _reshape_close_only_wide(raw_df)
    _validate_input(raw_df)

    raw_df["date"] = pd.to_datetime(raw_df["date"], errors="coerce")
    raw_df = raw_df.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"])

    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        raw_df[col] = pd.to_numeric(raw_df[col], errors="coerce")
    raw_df = raw_df.dropna(subset=numeric_cols)

    if "ticker" not in raw_df.columns:
        if ticker is None:
            ticker = input_path.stem.split("_")[0].upper()
        raw_df["ticker"] = ticker
    elif ticker is not None:
        raw_df["ticker"] = ticker

    processed_df = _engineer_features(raw_df)
    processed_df = _create_targets(processed_df, prediction_horizon)
    processed_df = processed_df.dropna().reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(output_path, index=False)

    splits = _temporal_split(processed_df, train_ratio=train_ratio, val_ratio=val_ratio)
    split_dir = output_path.parent / f"{output_path.stem}_splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    split_paths: Dict[str, str] = {}
    for split_name, split_df in splits.items():
        split_path = split_dir / f"{split_name}.csv"
        split_df.to_csv(split_path, index=False)
        split_paths[split_name] = str(split_path)

    metadata = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "ticker": ticker,
        "rows_in": int(len(raw_df)),
        "rows_out": int(len(processed_df)),
        "prediction_horizon": int(prediction_horizon),
        "feature_columns": [
            c for c in processed_df.columns if c not in {"date", "ticker", "target_close", "target_return", "target_direction"}
        ],
        "target_columns": ["target_close", "target_return", "target_direction"],
        "split_paths": split_paths,
    }
    metadata_path = output_path.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "processed_file": str(output_path),
        "metadata_file": str(metadata_path),
        "train_file": split_paths["train"],
        "validation_file": split_paths["validation"],
        "test_file": split_paths["test"],
    }


def main() -> None:
    args = parse_args()
    outputs = process_file(
        input_path=Path(args.input),
        output_path=Path(args.output),
        ticker=args.ticker,
        prediction_horizon=args.prediction_horizon,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    print("Processing complete.")
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
