#!/usr/bin/env bash
# Packages this repo's git-tracked files into a single zip for manual upload
# to Zenodo (zenodo.org/uploads/new), bypassing GitHub's release-archive path
# entirely — see README/MEMORY notes on the "Include Git LFS objects in
# archives" toggle being unreliable for this repo.
#
# Usage: tools/package_for_zenodo.sh [version-label]
#   version-label defaults to the current git tag, or DATE-COMMIT if untagged.
# Output dir override: OUT_DIR=/some/path tools/package_for_zenodo.sh
set -euo pipefail

cd "$(dirname "$0")/.."

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "[error] not inside a git repo" >&2
  exit 1
fi

if ! command -v zip >/dev/null 2>&1; then
  echo "[error] 'zip' not installed" >&2
  exit 1
fi

version="${1:-$(git describe --tags 2>/dev/null || echo "$(date +%Y%m%d)-$(git rev-parse --short HEAD)")}"
out_dir="${OUT_DIR:-$HOME/zenodo_uploads}"
out_name="oilspill-detection-dataset_${version}.zip"
out_path="${out_dir}/${out_name}"

mkdir -p "$out_dir"

echo "[info] checking tracked binary files for unresolved Git LFS pointers..."
pointer_hits=0
while IFS= read -r -d '' f; do
  case "$f" in
    *.tif|*.tiff|*.png|*.gpkg|*.geojson)
      size=$(stat -c%s "$f" 2>/dev/null || echo 0)
      if [[ "$size" -lt 512 ]] && head -c 20 "$f" 2>/dev/null | grep -q "^version https://git-lfs"; then
        echo "  [warn] $f is still an LFS pointer ($size bytes)" >&2
        pointer_hits=$((pointer_hits + 1))
      fi
      ;;
  esac
done < <(git ls-files -z)

if [[ "$pointer_hits" -gt 0 ]]; then
  echo "[error] $pointer_hits tracked file(s) are unresolved LFS pointers." >&2
  echo "        Run 'git lfs pull' in this repo first, then retry." >&2
  exit 1
fi

file_count=$(git ls-files | wc -l)
echo "[info] packaging ${file_count} tracked files -> ${out_path}"
rm -f "$out_path"
git ls-files -z | xargs -0 zip -q -X "$out_path"

echo "[done] $(du -h "$out_path" | cut -f1)  ->  ${out_path}"
echo "[info] upload this file at https://zenodo.org/uploads/new"
