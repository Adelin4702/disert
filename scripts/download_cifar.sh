#!/usr/bin/env bash
# CIFAR downloader using the fast.ai AWS-S3 mirrors, which (unlike
# cs.toronto.edu) support HTTP range/resume and stay up on throttled networks.
# Extracts to an ImageFolder layout that src/data.py auto-detects:
#   data/<dataset>/{train,test}/<class>/*.png
#
# Usage: bash scripts/download_cifar.sh [cifar10|cifar100]
set -u
DATASET="${1:-cifar10}"
case "$DATASET" in
  cifar10)  URL="https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz";  SIZE=135107811 ;;
  cifar100) URL="https://s3.amazonaws.com/fast-ai-imageclas/cifar100.tgz"; SIZE=169168619 ;;
  *) echo "usage: $0 [cifar10|cifar100]"; exit 2 ;;
esac
TGZ="data/${DATASET}.tgz"
mkdir -p data

if [ -d "data/${DATASET}/train" ] && [ -d "data/${DATASET}/test" ]; then
  echo "OK: data/${DATASET} already extracted"; exit 0
fi

i=0
until [ -f "$TGZ" ] && [ "$(stat -f%z "$TGZ" 2>/dev/null)" -ge "$SIZE" ]; do
  i=$((i+1)); [ "$i" -gt 200 ] && { echo "FAILED after $i attempts"; exit 1; }
  echo "attempt $i: have $(( $(stat -f%z "$TGZ" 2>/dev/null || echo 0)/1024/1024 )) MB, resuming..."
  curl -L -C - --connect-timeout 15 --max-time 180 --speed-limit 3000 --speed-time 30 \
       -o "$TGZ" "$URL" 2>/dev/null
done

echo "extracting..."
tar xzf "$TGZ" -C data/
echo "OK: train=$(find data/${DATASET}/train -name '*.png' | wc -l) test=$(find data/${DATASET}/test -name '*.png' | wc -l)"
