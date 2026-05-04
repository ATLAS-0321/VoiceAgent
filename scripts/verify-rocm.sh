#!/usr/bin/env bash
# Pre-flight check: ROCm + PyTorch + GPU detected
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/env.sh"

if [ -z "${VIRTUAL_ENV:-}" ]; then
    echo "Activating venv..."
    # shellcheck disable=SC1091
    source "$ROOT/.venv/bin/activate"
fi

echo "================================================================"
echo "  ROCm / PyTorch verification"
echo "================================================================"

echo ""
echo "[ROCm version]"
if command -v rocminfo >/dev/null 2>&1; then
    rocminfo | grep -iE "version|gfx" | head -5
else
    echo "  rocminfo not found"
    exit 1
fi

echo ""
echo "[PyTorch]"
python - <<'PY'
import torch
print(f"  torch:       {torch.__version__}")
print(f"  torch.hip:   {getattr(torch.version, 'hip', None)}")
print(f"  cuda avail:  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  device:      {torch.cuda.get_device_name(0)}")
    print(f"  vram total:  {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
PY

echo ""
echo "[Test GPU compute]"
python - <<'PY'
import torch
if not torch.cuda.is_available():
    print("  SKIP - no GPU"); raise SystemExit(1)
a = torch.randn(2048, 2048, device="cuda")
b = torch.randn(2048, 2048, device="cuda")
torch.cuda.synchronize()
import time
t0 = time.perf_counter()
for _ in range(10):
    c = a @ b
torch.cuda.synchronize()
dt = (time.perf_counter() - t0) / 10 * 1000
print(f"  matmul 2048x2048 avg: {dt:.2f} ms")
PY

echo ""
echo "  All checks passed."
