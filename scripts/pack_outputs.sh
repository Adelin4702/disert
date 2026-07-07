#!/usr/bin/env bash
# Bundle results + plots + checkpoints into one archive for easy download
# off a RunPod pod. Run from the repo root:  bash scripts/pack_outputs.sh
set -euo pipefail
STAMP="${1:-run}"
OUT="outputs_${STAMP}.tar.gz"

# checkpoints can be large; pass "--no-ckpt" to skip them
INCLUDE_CKPT=1
[ "${2:-}" = "--no-ckpt" ] && INCLUDE_CKPT=0

PATHS=(results plots)
[ "$INCLUDE_CKPT" = "1" ] && PATHS+=(checkpoints)

tar czf "$OUT" "${PATHS[@]}" 2>/dev/null || tar czf "$OUT" \
  $(for p in "${PATHS[@]}"; do [ -e "$p" ] && echo "$p"; done)

echo "wrote $OUT ($(du -h "$OUT" | cut -f1))"
echo "download it with:  runpodctl send $OUT     (then 'runpodctl receive <code>' on your laptop)"
echo "or via the Jupyter file browser (right-click -> Download)."
