# AI Prompt Log Template

> Fill one row per meaningful AI interaction (Grok/Claude/GPT/Cursor/etc.).

| # | Date | Tool | Stage | Prompt (short) | Output Summary | How Used | Verification |
|---|---|---|---|---|---|---|---|
| 1 | 2026-..-.. | GPT | Model design | "Design conditional diffusion policy for dual-arm action chunking." | Suggested Transformer + DDPM loss | Adopted architecture draft | Code ran on smoke test |
| 2 | 2026-..-.. | GPT | Debugging | "Fix torch.load error after PyTorch 2.6 weights_only change." | Use `weights_only=False` for trusted checkpoints | Patched eval/sample scripts | Evaluation script runs |
| 3 | 2026-..-.. | GPT | Experimentation | "Generate ablation runner for history/horizon/diffusion/demo count." | Produced automation script outline | Implemented `run_ablations.py` | Summary CSV generated |

## Notes

- Keep prompts concise but specific.
- Include failed prompts and what was learned.
- For grading, emphasize how AI accelerated implementation, debugging, and experiment analysis.

## Team Handoff Context

When this log is used in the final team report, tag each row with owner scope in `Output Summary` or `How Used`:
- `S1` environment/data pipeline
- `S2` BC/DAgger/covariate shift
- `S3` diffusion model/training/sampling
- `S4` integration/analysis/reporting

This prevents cross-student attribution drift during final write-up.

