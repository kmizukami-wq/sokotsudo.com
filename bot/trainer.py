#!/usr/bin/env python3
"""
Walk-forward trainer for XGBoost signal confidence model.

Uses purged walk-forward validation to prevent overfitting.
Train on 6 months, embargo 1 week, test on 2 months, roll forward.

Usage:
    python trainer.py                      # Walk-forward with 3 years data
    python trainer.py --days 730           # Custom period
    python trainer.py --retrain            # Weekly retrain (production)
"""

import sys
import math
import json
import argparse
import numpy as np
from datetime import datetime, timezone, timedelta
from urllib.request import Request, urlopen

from ml_model import FeatureBuilder, SignalPredictor, features_to_matrix

BB_BASE = "https://public.bitbank.cc"


def fetch_hourly_data(days=365 * 3):
    """Fetch hourly BTC/JPY OHLCV from bitbank."""
    all_candles = []
    today = datetime.now(timezone.utc)

    print(f"Fetching {days} days of hourly data from bitbank...")

    for d in range(days, -1, -1):
        date = today - timedelta(days=d)
        date_str = date.strftime("%Y%m%d")
        url = f"{BB_BASE}/btc_jpy/candlestick/1hour/{date_str}"
        try:
            req = Request(url)
            with urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
            ohlcv = data["data"]["candlestick"][0]["ohlcv"]
            for row in ohlcv:
                all_candles.append({
                    "open": float(row[0]),
                    "high": float(row[1]),
                    "low": float(row[2]),
                    "close": float(row[3]),
                    "volume": float(row[4]),
                    "timestamp": int(row[5]),
                })
        except Exception as e:
            if d % 30 == 0:
                print(f"  Warning: failed {date_str}: {e}")
            continue

        if d % 90 == 0:
            print(f"  Fetched up to {date_str} ({len(all_candles)} candles)")

    all_candles.sort(key=lambda c: c["timestamp"])
    n = len(all_candles)
    print(f"Total: {n} hourly candles ({n/24:.0f} days)")

    return {
        "open": np.array([c["open"] for c in all_candles]),
        "high": np.array([c["high"] for c in all_candles]),
        "low": np.array([c["low"] for c in all_candles]),
        "close": np.array([c["close"] for c in all_candles]),
        "volume": np.array([c["volume"] for c in all_candles]),
        "timestamps": np.array([c["timestamp"] for c in all_candles]),
    }


def walk_forward_validate(data, train_hours=24*180, test_hours=24*60,
                          embargo_hours=24*7, label_horizon=5,
                          label_threshold=0.01):
    """Walk-forward validation with embargo.

    Args:
        data: dict with close, high, low, volume, timestamps arrays
        train_hours: training window size (default 6 months)
        test_hours: test window size (default 2 months)
        embargo_hours: gap between train and test (default 1 week)
        label_horizon: bars ahead for label
        label_threshold: min move for +1/-1 label

    Returns:
        dict with results and the last trained model
    """
    closes = data["close"]
    highs = data["high"]
    lows = data["low"]
    volumes = data["volume"]
    timestamps = data["timestamps"]
    n = len(closes)

    fb = FeatureBuilder()

    # Build features and labels for entire dataset
    print("Building features...")
    features = fb.build(closes, highs, lows, volumes, timestamps)
    labels = fb.build_labels(closes, horizon=label_horizon,
                             threshold=label_threshold)
    X, valid_mask = features_to_matrix(features, fb.feature_names)

    # Combined mask: valid features AND valid labels
    label_valid = np.isfinite(labels)
    full_mask = valid_mask & label_valid

    print(f"Total bars: {n}, Valid for training: {np.sum(full_mask)}")

    # Walk-forward folds
    fold_results = []
    step = test_hours  # Roll forward by test window size
    start = train_hours

    fold_num = 0
    last_model = None

    while start + embargo_hours + test_hours <= n:
        fold_num += 1
        train_start = max(0, start - train_hours)
        train_end = start
        test_start = start + embargo_hours
        test_end = min(test_start + test_hours, n)

        # Train mask
        train_idx = np.arange(train_start, train_end)
        train_mask = full_mask[train_idx]
        X_train = X[train_idx[train_mask]]
        y_train = labels[train_idx[train_mask]]

        # Test mask
        test_idx = np.arange(test_start, test_end)
        test_mask = full_mask[test_idx]
        X_test = X[test_idx[test_mask]]
        y_test = labels[test_idx[test_mask]]

        if len(X_train) < 100 or len(X_test) < 20:
            print(f"  Fold {fold_num}: insufficient data "
                  f"(train={len(X_train)}, test={len(X_test)}), skipping")
            start += step
            continue

        # Train
        predictor = SignalPredictor()
        predictor.train(X_train, y_train)
        last_model = predictor

        # Evaluate on test set
        y_mapped = y_test + 1  # -1→0, 0→1, 1→2
        probs = predictor.model.predict_proba(X_test)
        preds = np.argmax(probs, axis=1)

        accuracy = np.mean(preds == y_mapped)

        # Direction accuracy: for +1 labels, does model predict +1?
        up_mask = y_test == 1
        down_mask = y_test == -1
        up_acc = np.mean(preds[up_mask] == 2) if np.any(up_mask) else 0
        down_acc = np.mean(preds[down_mask] == 0) if np.any(down_mask) else 0

        # Confidence-filtered accuracy (prob > 0.5 for direction)
        confident_correct = 0
        confident_total = 0
        for j in range(len(X_test)):
            if y_test[j] == 1 and probs[j, 2] > 0.5:
                confident_total += 1
                if preds[j] == 2:
                    confident_correct += 1
            elif y_test[j] == -1 and probs[j, 0] > 0.5:
                confident_total += 1
                if preds[j] == 0:
                    confident_correct += 1

        conf_acc = confident_correct / confident_total if confident_total > 0 else 0

        fold_result = {
            "fold": fold_num,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "accuracy": accuracy,
            "up_accuracy": up_acc,
            "down_accuracy": down_acc,
            "confident_accuracy": conf_acc,
            "confident_trades": confident_total,
        }
        fold_results.append(fold_result)

        print(f"  Fold {fold_num}: train={len(X_train):>5d}, test={len(X_test):>4d}, "
              f"acc={accuracy:.3f}, up_acc={up_acc:.3f}, "
              f"down_acc={down_acc:.3f}, conf_acc={conf_acc:.3f} "
              f"({confident_total} trades)")

        start += step

    # Summary
    if fold_results:
        avg_acc = np.mean([r["accuracy"] for r in fold_results])
        avg_up = np.mean([r["up_accuracy"] for r in fold_results])
        avg_down = np.mean([r["down_accuracy"] for r in fold_results])
        avg_conf = np.mean([r["confident_accuracy"] for r in fold_results])
        total_conf_trades = sum(r["confident_trades"] for r in fold_results)

        print(f"\n{'='*60}")
        print(f"  WALK-FORWARD RESULTS ({fold_num} folds)")
        print(f"{'='*60}")
        print(f"  Avg Accuracy:        {avg_acc:.3f}")
        print(f"  Avg Up Accuracy:     {avg_up:.3f}")
        print(f"  Avg Down Accuracy:   {avg_down:.3f}")
        print(f"  Avg Confident Acc:   {avg_conf:.3f}")
        print(f"  Total Conf Trades:   {total_conf_trades}")
        print(f"{'='*60}")

        # Feature importance from last model
        if last_model:
            print("\nFeature Importance (top 10):")
            imp = last_model.feature_importance()
            for i, (name, score) in enumerate(imp.items()):
                if i >= 10:
                    break
                print(f"  {i+1:2d}. {name:<25s} {score:.4f}")

    return {
        "folds": fold_results,
        "model": last_model,
    }


def retrain_latest(days=180):
    """Retrain model on latest data and save (for production use)."""
    data = fetch_hourly_data(days)
    if len(data["close"]) < 24 * 30:
        print("ERROR: Not enough data for training")
        sys.exit(1)

    fb = FeatureBuilder()
    features = fb.build(data["close"], data["high"], data["low"],
                        data["volume"], data["timestamps"])
    labels = fb.build_labels(data["close"], horizon=5, threshold=0.01)
    X, valid_mask = features_to_matrix(features, fb.feature_names)

    label_valid = np.isfinite(labels)
    full_mask = valid_mask & label_valid

    X_train = X[full_mask]
    y_train = labels[full_mask]

    print(f"Training on {len(X_train)} samples...")
    predictor = SignalPredictor()
    predictor.train(X_train, y_train)
    predictor.save()
    print(f"Model saved to {predictor.model_path}")

    # Show feature importance
    print("\nFeature Importance:")
    imp = predictor.feature_importance()
    for i, (name, score) in enumerate(imp.items()):
        if i >= 10:
            break
        print(f"  {i+1:2d}. {name:<25s} {score:.4f}")


def main():
    parser = argparse.ArgumentParser(description="XGBoost Walk-Forward Trainer")
    parser.add_argument("--days", type=int, default=365 * 3,
                        help="Days of data for walk-forward")
    parser.add_argument("--retrain", action="store_true",
                        help="Retrain on latest 6 months and save model")
    parser.add_argument("--train-months", type=int, default=6,
                        help="Training window in months")
    parser.add_argument("--test-months", type=int, default=2,
                        help="Test window in months")
    args = parser.parse_args()

    if args.retrain:
        retrain_latest(days=args.train_months * 30)
        return

    data = fetch_hourly_data(args.days)
    if len(data["close"]) < 100:
        print("ERROR: Not enough data fetched.")
        sys.exit(1)

    results = walk_forward_validate(
        data,
        train_hours=24 * args.train_months * 30,
        test_hours=24 * args.test_months * 30,
    )

    # Save the last model
    if results["model"]:
        results["model"].save()
        print(f"\nLast fold model saved to {results['model'].model_path}")


if __name__ == "__main__":
    main()
