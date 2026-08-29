#!/usr/bin/env bash
set -euo pipefail

msg="${1:-}"
pathspec="${2:-.}"

if [[ -z "$msg" ]]; then
	echo "Usage: $0 \"commit message\" [pathspec]"
	exit 2
fi

# Bootstrap: unlike src/do_git.sh, this repo may not be `git init`'d yet on
# first run (the dataset repo is LFS-backed, so init/LFS setup happens here
# instead of ahead of time).
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
	git init
fi

if ! command -v git-lfs >/dev/null 2>&1; then
	echo "[error] git-lfs not installed — install it first (e.g. apt install git-lfs)"
	exit 2
fi

git lfs install --local

# Configure remote without embedding credentials in the URL.
git_user="${GIT_USER:-Garcia-INPE}"
git_prj="${GIT_PRJ:-oilspill-detection-dataset}"
origin_url="https://github.com/${git_user}/${git_prj}.git"

if git remote get-url origin >/dev/null 2>&1; then
	git remote set-url origin "$origin_url"
else
	git remote add origin "$origin_url"
fi

git add "$pathspec"

if git diff --cached --quiet; then
	echo "[info] nothing to commit"
	exit 0
fi

git commit -m "$msg"

branch="$(git rev-parse --abbrev-ref HEAD)"
git push -u origin "$branch"

echo "[done] pushed ${branch} to origin"
