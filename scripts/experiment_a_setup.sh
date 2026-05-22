#!/usr/bin/env bash
# Experiment A — stage an isolated, faithful checkout of the 17-stage
# deep run's exact code + config (commit 6eb0566, rt 0.2.5) for
# inspection. Does NOT train: training is a Colab/GPU job (see
# docs/experiment_a_faithful_repro.md). This only creates a git
# worktree and prints the provenance so the economy / roster /
# curriculum can be verified before committing GPU hours.
#
# Usage:  scripts/experiment_a_setup.sh [worktree_dir]
# Default worktree_dir: ../reinforce-tactics-expA
set -euo pipefail

DEEP_COMMIT="6eb0566639ab18c1d40f99af80cfbaa71bc57c28"
WORKTREE="${1:-../reinforce-tactics-expA}"

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

if ! git cat-file -e "${DEEP_COMMIT}^{commit}" 2>/dev/null; then
  echo "ERROR: commit ${DEEP_COMMIT} not found. Fetch full history first:" >&2
  echo "  git fetch --unshallow   # or: git fetch origin ${DEEP_COMMIT}" >&2
  exit 1
fi

if [ -e "$WORKTREE" ]; then
  echo "ERROR: $WORKTREE already exists. Remove it or pass another path." >&2
  echo "  git worktree remove $WORKTREE" >&2
  exit 1
fi

echo "Creating detached worktree at ${DEEP_COMMIT:0:7} -> $WORKTREE"
git worktree add --detach "$WORKTREE" "$DEEP_COMMIT"

echo
echo "=== Provenance of the faithful tree (verify before training) ==="
git -C "$WORKTREE" log -1 --format='commit : %H%ndate   : %ci%nsubject: %s'
echo
echo "economy (constants.py):"
grep -E '^STARTING_GOLD|^HEADQUARTERS_INCOME' "$WORKTREE/reinforcetactics/constants.py" | sed 's/^/  /'
awk '/"W": \{/{f=1} f&&/"attack"/{print "  Warrior "$0; f=0}' "$WORKTREE/reinforcetactics/constants.py"
awk '/"K": \{/{f=1} f&&/"defence"/{print "  Knight  "$0; f=0}' "$WORKTREE/reinforcetactics/constants.py"
echo
echo "config (configs/ppo/bootstrap.yaml):"
grep -nE 'turn_penalty:' "$WORKTREE/configs/ppo/bootstrap.yaml" | head -1 | sed 's/^/  /'
echo "  curriculum stages: $(grep -cE '^\s+- name:' "$WORKTREE/configs/ppo/bootstrap.yaml")"
echo "  enabled_units in YAML: $(grep -c 'enabled_units' "$WORKTREE/configs/ppo/bootstrap.yaml" || true) (expected 0 -- roster is a notebook override)"
echo
echo "roster override (notebooks/ppo_bootstrap.ipynb cell 11):"
python3 - "$WORKTREE/notebooks/ppo_bootstrap.ipynb" <<'PY'
import json, sys
nb = json.load(open(sys.argv[1]))
for c in nb["cells"]:
    s = "".join(c["source"])
    for ln in s.splitlines():
        st = ln.strip()
        if "enabled_units" in st and "apply_overrides" in st and not st.startswith("#"):
            print("  ACTIVE:", st[:90])
PY
echo
echo "Next: run notebooks/ppo_bootstrap.ipynb FROM THIS WORKTREE on a"
echo "GPU/Colab runtime, unedited. See docs/experiment_a_faithful_repro.md"
echo "for the Colab procedure and the decision criteria."
echo
echo "Teardown when done:  git worktree remove $WORKTREE"
