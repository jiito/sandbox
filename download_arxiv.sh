#!/usr/bin/env bash

set -euo pipefail

# Usage: download_arxiv.sh <pdf_url_or_arxiv_id> [output_path]
# Examples:
#   download_arxiv.sh https://arxiv.org/pdf/2406.07882
#   download_arxiv.sh 2406.07882 /path/to/out.pdf

input="$1"
output="${2:-}"

# Normalize arXiv id to PDF URL if needed
if [[ "$input" =~ ^https?:// ]]; then
  url="$input"
else
  url="https://arxiv.org/pdf/${input}"
fi

# Derive default output filename if not provided
if [[ -z "$output" ]]; then
  # extract filename from URL
  fname="$(basename "$url")"
  # ensure .pdf
  case "$fname" in
    *.pdf) output="$PWD/$fname" ;;
    *) output="$PWD/${fname}.pdf" ;;
  esac
fi

# Create target dir if needed
mkdir -p "$(dirname "$output")"

# Temporary file for download
tmpfile="${output}.part"

# Retry download with curl
echo "Downloading $url -> $output"
if curl --fail --location --show-error --retry 3 --retry-delay 2 --output "$tmpfile" "$url"; then
  mv -- "$tmpfile" "$output"
  echo "Saved to $output"
  exit 0
else
  rm -f -- "$tmpfile"
  echo "Download failed: $url" >&2
  exit 2
fi
