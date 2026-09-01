# Repository rulesets

Source-of-truth JSON for branch rulesets applied to `pytorch/rl`.

These files are **not** auto-applied by GitHub. They live here so the configuration
is reviewable, version-controlled, and reproducible. To apply a ruleset:

```bash
gh api -X POST repos/pytorch/rl/rulesets --input .github/rulesets/<file>.json
```

To update an existing ruleset, fetch its id and `PUT`:

```bash
gh api repos/pytorch/rl/rulesets -q '.[] | select(.name=="<name>") | .id'
gh api -X PUT repos/pytorch/rl/rulesets/<id> --input .github/rulesets/<file>.json
```

To delete:

```bash
gh api -X DELETE repos/pytorch/rl/rulesets/<id>
```

## Current rulesets

- `lint-required.json` — gates merges into `main` / `nightly` / `release/*` on
  the `lint-done` aggregator from `.github/workflows/lint.yml` and the existing
  `Meta CLA Check`.
