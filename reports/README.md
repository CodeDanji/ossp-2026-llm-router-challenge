# PromptBudget reports

These reports contain only aggregate measurements from the public Train and Dev material.
They never contain a prompt, an episode ID, a model answer, or an outcome row.

Regenerate the selected-policy comparison after materializing the official public data:

```console
PYTHONPATH=src python tools/evaluate_policy.py \
  --input data/materialized/dev/inputs.json \
  --outcomes data/dev/outcomes.json \
  --artifact artifacts/promptbudget-v1/artifact.json \
  --manifest artifacts/promptbudget-v1/manifest.json \
  --report reports/dev_policy_comparison.json \
  --markdown-report reports/dev_policy_comparison.md
```
