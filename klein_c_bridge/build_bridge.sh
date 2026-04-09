#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KLEIN_C_DIR="$ROOT_DIR/../klein-cuda-c"
OUT_SO="$ROOT_DIR/klein_c_bridge/libklein_bridge.so"

if [[ ! -d "$KLEIN_C_DIR" ]]; then
  echo "Missing klein-cuda-c directory at: $KLEIN_C_DIR" >&2
  exit 1
fi

make -C "$KLEIN_C_DIR" cuda

cc -shared -fPIC -O3 \
  -I"$KLEIN_C_DIR" \
  "$ROOT_DIR/klein_c_bridge/bridge.c" \
  "$KLEIN_C_DIR"/flux.cuda.o \
  "$KLEIN_C_DIR"/flux_kernels.cuda.o \
  "$KLEIN_C_DIR"/flux_tokenizer.cuda.o \
  "$KLEIN_C_DIR"/flux_vae.cuda.o \
  "$KLEIN_C_DIR"/flux_transformer.cuda.o \
  "$KLEIN_C_DIR"/flux_sample.cuda.o \
  "$KLEIN_C_DIR"/flux_image.cuda.o \
  "$KLEIN_C_DIR"/jpeg.cuda.o \
  "$KLEIN_C_DIR"/flux_safetensors.cuda.o \
  "$KLEIN_C_DIR"/flux_qwen3.cuda.o \
  "$KLEIN_C_DIR"/flux_qwen3_tokenizer.cuda.o \
  "$KLEIN_C_DIR"/terminals.cuda.o \
  "$KLEIN_C_DIR"/flux_cuda.o \
  -L/usr/local/cuda/lib64 -lcudart -lcublas -lopenblas -lstdc++ -lm \
  -o "$OUT_SO"

echo "Built $OUT_SO"
