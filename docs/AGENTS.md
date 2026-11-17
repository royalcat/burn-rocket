# AGENTS.md — docs/

Layout and rules for the documentation tree. Repo-wide notes are in the root
[`AGENTS.md`](../AGENTS.md).

| path | audience | contents |
|---|---|---|
| `recipes/` | humans deploying a model | short pages: what the model is, the commands for each profile, limits, links to the details |
| `experiments/` | developers and agents | dated measurements, design notes, verification gates, board command lists, per-phase write-ups |

## Rules

- **Recipes stay short and plain.** Commands, a sentence on what each flag buys, a few
  headline numbers. No dated measurement tables, phase numbers, board protocols or
  verification gates; link to `experiments/` instead.
- **Detail goes to `experiments/`** (or the aligned `AGENTS.md`): measurements, parity
  tables, board commands, memory rounds, rejected ideas.
- Experiment files are historical records. Keep the flags and paths as they were run
  (the older logs use pre-2026-10-31 flag names; `experiments/README.md` maps them).
- When a measurement changes the headline numbers in the root `README.md` or a recipe,
  update both.

## Moves (2026-11-17)

- `docs/experiment-log*.md` -> `docs/experiments/`.
- `recipes/clef-oracle.md`, `clef-pack.md`, `clef-phase0.md` -> `experiments/`.
- The previous long `recipes/clef.md` and `recipes/decision.md` are now
  `experiments/clef-omni.md` and `experiments/decision-d3.md`; the recipes of those names
  are new short pages. Section names quoted elsewhere ("NPU placement", "Board usage",
  "Deployment (current)", ...) refer to those experiments files.
- The measured parity-profile and board tables of `recipes/embeddinggemma2.md` ->
  `experiments/embeddinggemma2-board.md`.
