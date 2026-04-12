#!/bin/bash
DPYTHON="C:/Users/94167/.conda/envs/dp39/python.exe"
ROOT="C:/Users/94167/Desktop/diff"

"$DPYTHON" "$ROOT/train_diffusion_policy.py" \
  --data-dir "$ROOT/preprocessed" --obs-file liftpot_images.npy --action-file liftpot_actions.npy --stats-file stats.json \
  --out-dir "$ROOT/ablations_fast/history_4_seed123" \
  --history 4 --horizon 8 --diffusion-steps 100 --epochs 30 --batch-size 128 \
  --lr 3e-4 --weight-decay 1e-4 --grad-clip 1.0 --d-model 256 --nhead 8 --num-layers 6 --dropout 0.1 \
  --beta-start 1e-4 --beta-end 0.02 --val-ratio 0.1 --stride 1 --num-workers 0 --seed 123 --device cuda \
  > "$ROOT/ablations_fast/history_4_seed123.log" 2>&1 &
echo "history_4_seed123 PID: $!"

"$DPYTHON" "$ROOT/train_diffusion_policy.py" \
  --data-dir "$ROOT/preprocessed" --obs-file liftpot_images.npy --action-file liftpot_actions.npy --stats-file stats.json \
  --out-dir "$ROOT/ablations_fast/horizon_8_seed42" \
  --history 4 --horizon 8 --diffusion-steps 100 --epochs 30 --batch-size 128 \
  --lr 3e-4 --weight-decay 1e-4 --grad-clip 1.0 --d-model 256 --nhead 8 --num-layers 6 --dropout 0.1 \
  --beta-start 1e-4 --beta-end 0.02 --val-ratio 0.1 --stride 1 --num-workers 0 --seed 42 --device cuda \
  > "$ROOT/ablations_fast/horizon_8_seed42.log" 2>&1 &
echo "horizon_8_seed42 PID: $!"

"$DPYTHON" "$ROOT/train_diffusion_policy.py" \
  --data-dir "$ROOT/preprocessed" --obs-file liftpot_images.npy --action-file liftpot_actions.npy --stats-file stats.json \
  --out-dir "$ROOT/ablations_fast/horizon_8_seed123" \
  --history 4 --horizon 8 --diffusion-steps 100 --epochs 30 --batch-size 128 \
  --lr 3e-4 --weight-decay 1e-4 --grad-clip 1.0 --d-model 256 --nhead 8 --num-layers 6 --dropout 0.1 \
  --beta-start 1e-4 --beta-end 0.02 --val-ratio 0.1 --stride 1 --num-workers 0 --seed 123 --device cuda \
  > "$ROOT/ablations_fast/horizon_8_seed123.log" 2>&1 &
echo "horizon_8_seed123 PID: $!"

"$DPYTHON" "$ROOT/train_diffusion_policy.py" \
  --data-dir "$ROOT/preprocessed" --obs-file liftpot_images.npy --action-file liftpot_actions.npy --stats-file stats.json \
  --out-dir "$ROOT/ablations_fast/diffusion_steps_100_seed42" \
  --history 4 --horizon 8 --diffusion-steps 100 --epochs 30 --batch-size 128 \
  --lr 3e-4 --weight-decay 1e-4 --grad-clip 1.0 --d-model 256 --nhead 8 --num-layers 6 --dropout 0.1 \
  --beta-start 1e-4 --beta-end 0.02 --val-ratio 0.1 --stride 1 --num-workers 0 --seed 42 --device cuda \
  > "$ROOT/ablations_fast/diffusion_steps_100_seed42.log" 2>&1 &
echo "diffusion_steps_100_seed42 PID: $!"

"$DPYTHON" "$ROOT/train_diffusion_policy.py" \
  --data-dir "$ROOT/preprocessed" --obs-file liftpot_images.npy --action-file liftpot_actions.npy --stats-file stats.json \
  --out-dir "$ROOT/ablations_fast/diffusion_steps_100_seed123" \
  --history 4 --horizon 8 --diffusion-steps 100 --epochs 30 --batch-size 128 \
  --lr 3e-4 --weight-decay 1e-4 --grad-clip 1.0 --d-model 256 --nhead 8 --num-layers 6 --dropout 0.1 \
  --beta-start 1e-4 --beta-end 0.02 --val-ratio 0.1 --stride 1 --num-workers 0 --seed 123 --device cuda \
  > "$ROOT/ablations_fast/diffusion_steps_100_seed123.log" 2>&1 &
echo "diffusion_steps_100_seed123 PID: $!"

"$DPYTHON" "$ROOT/train_diffusion_policy.py" \
  --data-dir "$ROOT/preprocessed" --obs-file liftpot_images.npy --action-file liftpot_actions.npy --stats-file stats.json \
  --out-dir "$ROOT/ablations_fast/horizon_4_seed42" \
  --history 4 --horizon 4 --diffusion-steps 100 --epochs 30 --batch-size 128 \
  --lr 3e-4 --weight-decay 1e-4 --grad-clip 1.0 --d-model 256 --nhead 8 --num-layers 6 --dropout 0.1 \
  --beta-start 1e-4 --beta-end 0.02 --val-ratio 0.1 --stride 1 --num-workers 0 --seed 42 --device cuda \
  > "$ROOT/ablations_fast/horizon_4_seed42.log" 2>&1 &
echo "horizon_4_seed42 PID: $!"
