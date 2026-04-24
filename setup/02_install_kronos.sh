#!/bin/bash
# ============================================================
# PHASE 2: Download Kronos model and run inference test
# ============================================================

set -e
cd ~/kronos-trader
source ~/Trading/kronos-trader/venv/bin/activate

echo "📦 Cloning Kronos repo..."
git clone https://github.com/shiyu-coder/Kronos.git
cd Kronos
~/Trading/kronos-trader/venv/bin/pip install -r requirements.txt
cd ..

echo "🤖 Downloading Kronos-small model from HuggingFace..."
~/Trading/kronos-trader/venv/bin/python3 - <<'EOF'
from huggingface_hub import snapshot_download
import os

os.makedirs("models", exist_ok=True)

print("Downloading Kronos-Tokenizer-base...")
snapshot_download(
    repo_id="NeoQuasar/Kronos-Tokenizer-base",
    local_dir="models/Kronos-Tokenizer-base"
)

print("Downloading Kronos-small...")
snapshot_download(
    repo_id="NeoQuasar/Kronos-small",
    local_dir="models/Kronos-small"
)

print("✅ Models downloaded successfully!")
EOF

echo "🧪 Running inference smoke test..."
python3 - <<'EOF'
import sys
sys.path.insert(0, 'Kronos')
import pandas as pd
import numpy as np
from model import Kronos, KronosTokenizer, KronosPredictor

print("Loading tokenizer and model...")
tokenizer = KronosTokenizer.from_pretrained("models/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("models/Kronos-small")
predictor = KronosPredictor(model, tokenizer, max_context=512)

# Generate synthetic candle data for test
np.random.seed(42)
n = 200
price = 50000 + np.cumsum(np.random.randn(n) * 100)
df = pd.DataFrame({
    'open':   price,
    'high':   price * 1.002,
    'low':    price * 0.998,
    'close':  price + np.random.randn(n) * 50,
    'volume': np.random.uniform(1, 10, n)
})
timestamps = pd.date_range("2024-01-01", periods=n, freq="1h")
future_ts  = pd.date_range(timestamps[-1], periods=25, freq="1h")[1:]

x_df = df.iloc[:200]
x_ts = timestamps[:200]

print("Running prediction...")
pred = predictor.predict(
    df=x_df,
    x_timestamp=x_ts,
    y_timestamp=future_ts,
    pred_len=24,
    T=1.0,
    top_p=0.9,
    sample_count=3
)
print(f"✅ Smoke test passed! Predicted {len(pred)} candles.")
print(pred[['open','high','low','close']].head())
EOF

echo ""
echo "✅ Kronos is installed and working on your Pi!"
