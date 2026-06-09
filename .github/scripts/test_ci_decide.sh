#!/usr/bin/env bash
# Table-driven tests for ci_decide.sh.
# Run: bash .github/scripts/test_ci_decide.sh

set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
script="$here/ci_decide.sh"

fail=0
pass=0

assert_line() {
  local out="$1" expected="$2" name="$3"
  if printf '%s\n' "$out" | grep -Fxq "$expected"; then
    pass=$((pass+1))
  else
    echo "FAIL [$name] expected line: $expected"
    echo "--- output ---"
    printf '%s\n' "$out"
    echo "--------------"
    fail=$((fail+1))
  fi
}

# ---------------------------------------------------------------------------
# 1. push to main -> full run regardless of file flags
# ---------------------------------------------------------------------------
out=$(bash "$script" push refs/heads/main '[]' false false false false)
assert_line "$out" 'full=true'             'push main: full'
assert_line "$out" 'run_cpu_matrix=true'   'push main: cpu matrix'
assert_line "$out" 'shards_json=["1","2","3"]' 'push main: all shards'
assert_line "$out" 'run_olddeps=true'      'push main: olddeps'

# ---------------------------------------------------------------------------
# 2. push to release/* -> full
# ---------------------------------------------------------------------------
out=$(bash "$script" push refs/heads/release/0.12.0 '[]' false false false false)
assert_line "$out" 'full=true'             'push release: full'

# ---------------------------------------------------------------------------
# 3. PR touching transforms only -> shard 1 only, Python 3.12 only
# ---------------------------------------------------------------------------
out=$(bash "$script" pull_request refs/pull/1/merge '["Transforms"]' true false false false)
assert_line "$out" 'full=false'                'PR transforms: not full'
assert_line "$out" 'run_cpu_matrix=false'      'PR transforms: single py'
assert_line "$out" 'run_shard1=true'           'PR transforms: shard1'
assert_line "$out" 'run_shard2=false'          'PR transforms: no shard2'
assert_line "$out" 'run_shard3=false'          'PR transforms: no shard3'
assert_line "$out" 'shards_json=["1"]'         'PR transforms: shards_json'
assert_line "$out" 'run_olddeps=false'         'PR transforms: no olddeps'
assert_line "$out" 'run_stable=false'          'PR transforms: no stable'
assert_line "$out" 'run_distributed=false'     'PR transforms: no distributed'

# ---------------------------------------------------------------------------
# 4. PR touching envs + something else -> shard 2 + shard 3
# ---------------------------------------------------------------------------
out=$(bash "$script" pull_request refs/pull/1/merge '[]' false true true false)
assert_line "$out" 'run_shard1=false'          'PR envs+core: no shard1'
assert_line "$out" 'run_shard2=true'           'PR envs+core: shard2'
assert_line "$out" 'run_shard3=true'           'PR envs+core: shard3'
assert_line "$out" 'shards_json=["2","3"]'     'PR envs+core: shards_json'

# ---------------------------------------------------------------------------
# 5. PR with ciflow/full -> full matrix
# ---------------------------------------------------------------------------
out=$(bash "$script" pull_request refs/pull/1/merge '["ciflow/full"]' false false false false)
assert_line "$out" 'full=true'                 'PR ciflow/full: full'
assert_line "$out" 'run_cpu_matrix=true'       'PR ciflow/full: cpu matrix'
assert_line "$out" 'shards_json=["1","2","3"]' 'PR ciflow/full: all shards'
assert_line "$out" 'run_olddeps=true'          'PR ciflow/full: olddeps'
assert_line "$out" 'run_stable=true'           'PR ciflow/full: stable'
assert_line "$out" 'run_optdeps=true'          'PR ciflow/full: optdeps'
assert_line "$out" 'run_distributed=true'      'PR ciflow/full: distributed'

# ---------------------------------------------------------------------------
# 6. PR with ciflow/gpu but no file changes -> all shards, but not olddeps/stable
# ---------------------------------------------------------------------------
out=$(bash "$script" pull_request refs/pull/1/merge '["ciflow/gpu"]' false false false false)
assert_line "$out" 'run_shard1=true'           'PR ciflow/gpu: shard1'
assert_line "$out" 'run_shard2=true'           'PR ciflow/gpu: shard2'
assert_line "$out" 'run_shard3=true'           'PR ciflow/gpu: shard3'
assert_line "$out" 'run_olddeps=false'         'PR ciflow/gpu: no olddeps'
assert_line "$out" 'run_stable=false'          'PR ciflow/gpu: no stable'

# ---------------------------------------------------------------------------
# 7. PR with ciflow/cpu-matrix only -> full cpu matrix, shard fallback
# ---------------------------------------------------------------------------
out=$(bash "$script" pull_request refs/pull/1/merge '["ciflow/cpu-matrix"]' false false false false)
assert_line "$out" 'run_cpu_matrix=true'       'PR ciflow/cpu-matrix: cpu matrix'
assert_line "$out" 'full=false'                'PR ciflow/cpu-matrix: not full'
assert_line "$out" 'shards_json=["3"]'         'PR ciflow/cpu-matrix: fallback shard3'

# ---------------------------------------------------------------------------
# 8. PR touching distributed code -> distributed flag
# ---------------------------------------------------------------------------
out=$(bash "$script" pull_request refs/pull/1/merge '[]' false false true true)
assert_line "$out" 'run_distributed=true'      'PR distributed changed: distributed true'

# ---------------------------------------------------------------------------
# 9. PR with ciflow/distributed label, no file changes -> distributed true
# ---------------------------------------------------------------------------
out=$(bash "$script" pull_request refs/pull/1/merge '["ciflow/distributed"]' false false false false)
assert_line "$out" 'run_distributed=true'      'PR ciflow/distributed: distributed true'

# ---------------------------------------------------------------------------
# 10. PR with no matching files and no labels -> shard 3 fallback
# ---------------------------------------------------------------------------
out=$(bash "$script" pull_request refs/pull/1/merge '[]' false false false false)
assert_line "$out" 'shards_json=["3"]'         'PR empty: shard3 fallback'
assert_line "$out" 'run_shard3=true'           'PR empty: shard3 true'
assert_line "$out" 'full=false'                'PR empty: not full'

# ---------------------------------------------------------------------------
# 11. workflow_dispatch -> full
# ---------------------------------------------------------------------------
out=$(bash "$script" workflow_dispatch refs/heads/main '[]' false false false false)
assert_line "$out" 'full=true'                 'workflow_dispatch: full'

# ---------------------------------------------------------------------------
# 12. Label name that is a substring of another label should not false-match.
#      (e.g. "ciflow/optdeps-extra" must not match "ciflow/optdeps")
#      This guards the exact-quoted grep.
# ---------------------------------------------------------------------------
out=$(bash "$script" pull_request refs/pull/1/merge '["ciflow/optdeps-extra"]' false false false false)
assert_line "$out" 'run_optdeps=false'         'label substring safety'

echo
echo "passed: $pass, failed: $fail"
exit $fail
