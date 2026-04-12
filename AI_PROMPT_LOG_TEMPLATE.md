# AI Prompt Log (Detailed Template)

Use this file to document all meaningful AI-assisted work from project start to final report.

## 1. Purpose

This log is evidence of how AI was used in development, debugging, experiments, analysis, and writing.
It should allow another teammate to understand:
- what was asked,
- what AI returned,
- what was actually adopted,
- how correctness was verified,
- and what was rejected.

## 2. How To Fill This Log

- Record one entry per meaningful interaction, not per chat message.
- Keep prompts concise but specific enough to reproduce context.
- Always separate `AI suggestion` from `what was implemented`.
- Include failed or partially useful prompts; they show iteration quality.
- Add file paths and commit hashes when possible.
- Use consistent language and units (e.g., MSE, smoothness, success rate).

## 3. Stage Codes

Use these stage codes in each entry.

- `S0` Project setup and planning
- `S1` Environment setup and dependency resolution
- `S2` Data collection (expert scripted + noisy demos)
- `S3` Data preprocessing and dataset checks
- `S4` Baseline methods (classic BC / DAgger)
- `S5` Diffusion-policy architecture and loss design
- `S6` Training loop, checkpointing, sampling, evaluation
- `S7` Ablation automation and visualization
- `S8` Statistical analysis and interpretation
- `S9` Documentation and final report writing

## 4. Work Tags

Use one or more tags to indicate where this interaction contributed:

- `A`: environment, demos, preprocessing pipeline
- `B`: BC / DAgger / covariate shift
- `C`: diffusion-policy model and training
- `D`: experiment integration, visualization, stats, report

## 5. Master Interaction Table

Fill one row per meaningful interaction.

| ID | Date | Tool | Stage | Work Tag(s) | Goal | Prompt (short) | AI Suggestion Summary | Implemented Change | Files Touched | Verification Method | Verification Result | Outcome | Commit / Artifact |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 001 | 2026-..-.. | GPT | S5 | C | Define conditional diffusion policy for action chunks | "Design model for multi-step action prediction conditioned on observation history" | Suggested sequence backbone + diffusion denoiser + noise schedule | Adopted model skeleton and adjusted hidden size for memory budget | `diffusion_policy/model.py`, `diffusion_policy/diffusion.py` | Smoke train (`1 epoch`), forward-shape checks | Passed; loss decreases in smoke run | Adopted | `commit: ....` |
| 002 | 2026-..-.. | GPT | S6 | C | Fix checkpoint loading issue | "Fix torch.load error after PyTorch weights_only behavior change" | Suggested trusted-load path with explicit flag handling | Patched load paths in eval/sample scripts | `eval_diffusion_policy.py`, `sample_policy.py` | Run eval on existing checkpoint | Eval script executes without load error | Adopted | `commit: ....` |
| 003 | 2026-..-.. | GPT | S7 | D | Automate ablation sweep | "Create script to run history/horizon/diffusion-step/demo-count ablations" | Proposed argumentized runner + summary aggregation | Implemented runner and summary pipeline | `run_ablations.py`, `ablations_fast/summary.csv` | Re-run subset and compare aggregated rows | Rows complete and consistent across seeds | Adopted | `artifact: ablations_fast/summary.csv` |

## 6. Prompt-Level Detail Blocks

For any high-impact interaction (architecture, critical bug fix, experiment conclusion), add a detail block.

### Detail Block Template

- `Interaction ID`:
- `Date / Time`:
- `Tool / Model`:
- `Stage`:
- `Work Tag(s)`:
- `Why this prompt was needed`:
- `Full Prompt`:
- `Key AI Response`:
- `Decision`:
- `What was changed in code/docs`:
- `Verification steps`:
- `Risks or limitations`:
- `Follow-up action`:
- `Commit / artifacts`:

### Example Detail Block

- `Interaction ID`: 014
- `Date / Time`: 2026-..-.. ..:..
- `Tool / Model`: GPT
- `Stage`: S7
- `Work Tag(s)`: D
- `Why this prompt was needed`: Existing ablation outputs were incomplete and difficult to compare.
- `Full Prompt`: "Given current run folders, design a robust summary script that detects missing runs and outputs mean/std by setting."
- `Key AI Response`: Suggested parsing run names into factors, validating required seeds, and generating one tidy CSV.
- `Decision`: Adopted with minor naming changes to match project conventions.
- `What was changed in code/docs`: Updated runner and plotting flow; regenerated summary and figures.
- `Verification steps`: Checked expected row count and per-factor two-seed completeness.
- `Risks or limitations`: Historical folders may still contain stale runs not part of the final table.
- `Follow-up action`: Keep only canonical summary for report and mark archive folders explicitly.
- `Commit / artifacts`: `commit: ....`, `ablations_fast/summary.csv`, `ablations_fast/plots/*.png`

## 7. Failure / Rejection Log

Record prompts that produced wrong, unsafe, or low-quality suggestions.

| ID | Date | Stage | Prompt (short) | Why Rejected | What Was Done Instead | Impact on Progress |
|---|---|---|---|---|---|---|
| F01 | 2026-..-.. | S6 | "Use larger model width without changing batch size" | Exceeded memory budget and unstable training | Reduced width and tuned batch / grad steps | Delayed one run; final training stable |

## 8. Verification Checklist Per Stage

Use this checklist before final submission.

- `S1` Environment
  - Dependency versions recorded
  - Training/eval entrypoints runnable
- `S2-S3` Data
  - Demo generation reproducible
  - Preprocessing outputs shape/normalization validated
- `S4` Baselines
  - BC/DAgger metrics reproducible
  - Covariate shift analysis documented
- `S5-S6` Diffusion policy
  - Model assumptions documented
  - Training and sampling scripts validated on smoke + main runs
- `S7-S8` Experiments and analysis
  - Ablation table complete for declared seeds
  - Figures correspond to current summary artifacts
  - Statistical interpretation matches numeric results
- `S9` Reporting
  - Claims traceable to code, tables, and plots
  - AI contribution log linked to concrete commits/artifacts

## 9. Submission Snapshot

Fill this section once before submission.

- `Repository`:
- `Branch`:
- `Final commit hash`:
- `Main report file(s)`:
- `Core result table`:
- `Core figure directory`:
- `Total AI interactions logged`:
- `High-impact interactions (count)`:
- `Rejected/failed interactions (count)`:
- `Known limitations disclosed in report`:

## 10. Minimal Quality Standard

This log is considered complete only if:
- Every major implementation block (A/B/C/D) has at least one logged interaction.
- Every key figure/table in the report can be linked to at least one interaction and one artifact.
- At least one failed/rejected interaction is documented with corrective action.
- Final claims are backed by verifiable outputs (scripts, commits, or generated artifacts).
