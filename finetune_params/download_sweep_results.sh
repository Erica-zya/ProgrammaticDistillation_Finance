#!/usr/bin/env bash
# Run from repo root: bash finetune_test/download_sweep_results.sh
# Syncs sweep plots and result JSONs from Modal volume into finetune_test/.
# Overwrites local files with volume contents so you always have the newest; no duplicate filenames.
# If you see "No such file or directory", the remote path isn't on the volume yet—run a sweep first.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p finetune_test

# Remove old nested duplicates from previous script behavior (sweep_results/sweep_results, etc.)
for dir in sweep_results sweep_plots; do
  nested="finetune_test/$dir/$dir"
  if [ -d "$nested" ]; then
    [ -n "$(ls -A "$nested" 2>/dev/null)" ] && mv "$nested"/* "finetune_test/$dir/" 2>/dev/null || true
    rmdir "$nested" 2>/dev/null || rm -rf "$nested" 2>/dev/null || true
  fi
done

# Use parent as destination so we get finetune_test/sweep_plots/ and finetune_test/sweep_results/
# (otherwise Modal nests again and creates finetune_test/sweep_results/sweep_results/)
ok=0
if modal volume get --force finance-data outputs/sweep_plots finetune_test; then
  echo "Downloaded sweep_plots into finetune_test/sweep_plots"
  ok=1
else
  echo "Could not get outputs/sweep_plots (run a sweep first to generate plots)"
fi
if modal volume get --force finance-data outputs/sweep_results finetune_test; then
  echo "Downloaded sweep_results into finetune_test/sweep_results"
  ok=1
else
  echo "Could not get outputs/sweep_results (run a sweep first to generate JSONs)"
fi

if [ "$ok" -eq 1 ]; then
  echo "Done."
else
  echo "Nothing on volume yet. Run: modal run finetune_test/modal_train_lora_sweep.py"
  exit 1
fi
