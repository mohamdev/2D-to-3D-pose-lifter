#!/usr/bin/env bash
#
# random_half_move.sh — move half the files from SOURCE_DIR to DEST_DIR at random
#
# Usage:
#   ./random_half_move.sh /path/to/source /path/to/dest
# or edit SOURCE_DIR and DEST_DIR below and run without args.

set -euo pipefail
IFS=$'\n\t'

# ——— CONFIGURATION ———
# You can either set these here:
SOURCE_DIR="${1:-/path/to/source_folder}"
DEST_DIR="${2:-/path/to/output_folder}"
# ————————————————————

# sanity checks
if [[ ! -d "$SOURCE_DIR" ]]; then
  echo "Error: SOURCE_DIR not found: $SOURCE_DIR" >&2
  exit 1
fi

# create dest if needed
mkdir -p "$DEST_DIR"

# count files in SOURCE_DIR (not including subdirs)
total=$(find "$SOURCE_DIR" -maxdepth 1 -type f | wc -l)
if (( total == 0 )); then
  echo "No files found in $SOURCE_DIR" >&2
  exit 1
fi

# compute half (round up if odd)
half=$(( (total + 1) / 2 ))

echo "Found $total files, moving $half of them at random..."

# pipeline: list → shuffle → take half → move
find "$SOURCE_DIR" -maxdepth 1 -type f -print0 \
  | shuf -z \
  | head -z -n "$half" \
  | xargs -0 mv -t "$DEST_DIR"

echo "Done! $half files moved to $DEST_DIR."

