#!/usr/bin/env bash
# Resilient pretrained-weights downloader for flaky networks.
# Fetches the ImageNet weights torchvision needs into its hub cache so that
# building the models offline just loads them (no download at train time).
set -u
DEST="$HOME/.cache/torch/hub/checkpoints"
MAX_ATTEMPTS="${1:-300}"
mkdir -p "$DEST"

# url  filename
FILES=(
  "https://download.pytorch.org/models/efficientnet_v2_s-dd5fe13b.pth efficientnet_v2_s-dd5fe13b.pth"
  "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth mobilenet_v3_small-047dcff4.pth"
)

fetch() {
  local url="$1" out="$2"
  for i in $(seq 1 "$MAX_ATTEMPTS"); do
    # a valid torch checkpoint unzips-check: just ensure non-trivial size + torch.load works later
    sz=$(stat -f%z "$out" 2>/dev/null || echo 0)
    echo "  [$(basename "$out")] attempt $i: have $((sz/1024/1024)) MB"
    curl -L -C - --connect-timeout 15 --max-time 120 \
         --speed-limit 5000 --speed-time 20 -o "$out" "$url" 2>/dev/null
    # heuristic completion: size stops growing across a full clean transfer
    # (curl exits 0 only when the server-declared length is met)
    if [ $? -eq 0 ]; then echo "  [$(basename "$out")] curl reports complete"; return 0; fi
    sleep 1
  done
  return 1
}

for entry in "${FILES[@]}"; do
  set -- $entry
  url="$1"; name="$2"
  echo "== $name =="
  fetch "$url" "$DEST/$name" || { echo "FAILED: $name"; exit 1; }
done
echo "OK: weights in $DEST"
ls -la "$DEST"
