# Implementation Sequence for Codex

Execute in this order unless blocked:

1. Scaffold repo.
2. Add config dataclasses/schemas.
3. Implement baseline Transformer.
4. Implement `ComposedLinear` with tests.
5. Implement `TinyCoreLM` basis-only.
6. Add toy dataset and tokenizer stub.
7. Add training loop.
8. Add generation loop.
9. Add parameter/byte accounting.
10. Run baseline vs TinyCore toy comparison.
11. Add low-rank deltas.
12. Add recurrent mixer.
13. Add ablation runner.
14. Add manifest checkpoint save/load.
15. Add agent protocol TypeScript package.
16. Add repo tools.
17. Add patch engine.
18. Add local server.
19. Add VSCode shell.
20. Start TCMDL/native runtime only after Python checkpoint stabilises.

After each step, run tests and update a machine-readable progress file:

```json
{
  "completed_tasks": [],
  "blocked_tasks": [],
  "latest_metrics": {},
  "architecture_invariant_status": "preserved|violated|unknown"
}
```
