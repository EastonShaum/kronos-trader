"""
Kronos Inference Engine
Supports Kronos-small (24.7M) and Kronos-mini (4.1M) simultaneously.
Each model is loaded once and cached in _predictors.
"""
import sys
import os
import logging
import pandas as pd

# Add Kronos repo to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Kronos'))

logger = logging.getLogger(__name__)

# Per-model config
_BASE = os.path.join(os.path.dirname(__file__), '..', 'models')

MODEL_CONFIGS = {
    "small": {
        "model_path":     os.getenv("MODEL_PATH",     os.path.join(_BASE, "Kronos-small")),
        "tokenizer_path": os.getenv("TOKENIZER_PATH", os.path.join(_BASE, "Kronos-Tokenizer-base")),
        "max_context":    int(os.getenv("MAX_CONTEXT", 512)),
        "label":          "Kronos-small",
    },
    "mini": {
        "model_path":     os.path.join(_BASE, "Kronos-mini"),
        "tokenizer_path": os.path.join(_BASE, "Kronos-Tokenizer-2k"),
        "max_context":    2048,
        "label":          "Kronos-mini",
    },
}

_predictors: dict = {}  # keyed by model name


def load_model(model_name: str = "small"):
    """Load a Kronos model by name and cache it. Returns the predictor."""
    if model_name in _predictors:
        return _predictors[model_name]

    cfg = MODEL_CONFIGS.get(model_name)
    if cfg is None:
        raise ValueError(f"Unknown model: {model_name!r}. Choose 'small' or 'mini'.")

    from model import Kronos, KronosTokenizer, KronosPredictor

    logger.info(f"Loading {cfg['label']} from {cfg['model_path']} ...")
    tokenizer = KronosTokenizer.from_pretrained(cfg["tokenizer_path"])
    model     = Kronos.from_pretrained(cfg["model_path"])
    predictor = KronosPredictor(model, tokenizer, max_context=cfg["max_context"])
    _predictors[model_name] = predictor
    logger.info(f"✅ {cfg['label']} loaded (context={cfg['max_context']}).")
    return predictor


def run_prediction(
    ohlcv_df: pd.DataFrame,
    symbol: str,
    horizon_hours: int = 24,
    sample_count: int = 5,
    triggered_by: str = "scheduler",
    model_name: str = "small",
) -> dict:
    """
    Run a single Kronos model prediction on OHLCV data.
    Returns a dict including model_name and model_label.
    """
    predictor = load_model(model_name)
    cfg       = MODEL_CONFIGS[model_name]

    lookback = min(int(os.getenv("LOOKBACK_CANDLES", 400)), len(ohlcv_df))
    df_input = ohlcv_df.tail(lookback).copy()
    df_input = df_input[["open", "high", "low", "close", "volume"]].ffill()

    last_ts  = df_input.index[-1]
    freq_str = _infer_freq(df_input.index)
    x_timestamp = pd.Series(df_input.index)
    y_timestamp = pd.Series(pd.date_range(
        start=last_ts, periods=horizon_hours + 1, freq=freq_str
    )[1:])

    current_price = float(df_input["close"].iloc[-1])

    logger.info(
        f"Running {cfg['label']} for {symbol} | "
        f"lookback={lookback} | horizon={horizon_hours}h | samples={sample_count}"
    )

    pred_df = predictor.predict(
        df=df_input.reset_index(drop=True),
        x_timestamp=x_timestamp,
        y_timestamp=pd.Series(y_timestamp),
        pred_len=horizon_hours,
        T=1.0,
        top_p=0.9,
        sample_count=sample_count,
    )

    predicted_close = float(pred_df["close"].iloc[-1])
    predicted_high  = float(pred_df["high"].max())
    predicted_low   = float(pred_df["low"].min())
    confidence      = float((pred_df["close"] > current_price).mean())

    pred_df_out = pred_df.copy()
    pred_df_out.index = pd.to_datetime(y_timestamp).dt.strftime("%Y-%m-%dT%H:%M:%S")
    raw_output = pred_df_out[["open", "high", "low", "close"]].to_json()

    logger.info(
        f"✅ [{cfg['label']}] {symbol} | current={current_price:.4f} | "
        f"predicted_close={predicted_close:.4f} | confidence={confidence:.1%}"
    )

    return {
        "symbol":            symbol,
        "current_price":     current_price,
        "predicted_close":   predicted_close,
        "predicted_high":    predicted_high,
        "predicted_low":     predicted_low,
        "confidence":        confidence,
        "horizon_hours":     horizon_hours,
        "triggered_by":      triggered_by,
        "raw_output":        raw_output,
        "timestamps":        [str(ts) for ts in y_timestamp],
        "predicted_candles": pred_df[["open", "high", "low", "close"]].values.tolist(),
        "model_name":        model_name,
        "model_label":       cfg["label"],
    }


def run_dual_prediction(
    ohlcv_df: pd.DataFrame,
    symbol: str,
    horizon_hours: int = 24,
    sample_count: int = 5,
    triggered_by: str = "scheduler",
) -> dict:
    """
    Run both Kronos-small and Kronos-mini on the same OHLCV data.
    Returns a combined dict with each model's full result plus an agreement flag.
    """
    small = run_prediction(ohlcv_df, symbol, horizon_hours, sample_count, triggered_by, "small")
    mini  = run_prediction(ohlcv_df, symbol, horizon_hours, sample_count, triggered_by, "mini")

    # Agreement: both bullish (>0.5) or both bearish (<=0.5)
    small_bullish = small["confidence"] > 0.5
    mini_bullish  = mini["confidence"]  > 0.5
    agreement     = small_bullish == mini_bullish

    if agreement:
        direction = "Bullish" if small_bullish else "Bearish"
        logger.info(f"✅ {symbol}: Both models agree — {direction} "
                    f"(small={small['confidence']:.1%} mini={mini['confidence']:.1%})")
    else:
        logger.info(f"⚠️  {symbol}: Models disagree — "
                    f"small={'Bullish' if small_bullish else 'Bearish'} ({small['confidence']:.1%})  "
                    f"mini={'Bullish' if mini_bullish else 'Bearish'} ({mini['confidence']:.1%})")

    return {
        "symbol":    symbol,
        "small":     small,
        "mini":      mini,
        "agreement": agreement,
    }


def _infer_freq(index: pd.DatetimeIndex) -> str:
    if len(index) < 2:
        return "1h"
    delta = (index[-1] - index[-2]).total_seconds()
    if delta <= 300:     return "5min"
    elif delta <= 900:   return "15min"
    elif delta <= 3600:  return "1h"
    elif delta <= 86400: return "1D"
    return "1h"
