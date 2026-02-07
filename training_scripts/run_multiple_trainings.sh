#!/bin/bash

# Fixed parameters
PYTHON_CMD="python /home/carola/masterthesis/cleanrl/cleanrl/td3_continuous_action_jax.py"
ENV_ID="PouringEnvIsaac-v0"
SEED=42
SAVE_MODEL="--save-model"
BUFFER_SIZE=50000
TOTAL_TIMESTEPS=500000
CAPTURE_VIDEO="--capture_video"
OUTPUT_DIR="/home/carola/masterthesis/cleanrl/cleanrl/outputs/"

# Parameter variations
TARGET_LEVEL_WGTS=(1.0)
PT_CUP_WGTS=(15)
PT_FLOW_WGTS=(0)
PT_SPILL_WGTS=(0 -300 -600)
ACTION_COSTS=(-0.01)
JUG_RESTING_WGTS=(-0.0000001)
JUG_VELOCITY_WGTS=(0)

DISTANCE_WGT=0

EXPLORATION_NOISE=0.1
#INITIAL_EXPLORATION_NOISE=0.1
#MIN_EXPLORATION_NOISE=0.02

#EXPLORATION_WARMUP_STEPS=0

SIGNAL_NOISE=(0.25 0.5 0.75)
MIN_SIGNAL_NOISE=0.00
MAX_SIGNAL_NOISE=1000

TIME_PENALTY=0
#(-0.001 -0.005 -0.02 -0.03)

# Loop through all combinations
for target_level_wgt in "${TARGET_LEVEL_WGTS[@]}"; do
  for pt_cup_wgt in "${PT_CUP_WGTS[@]}"; do
    for pt_flow_wgt in "${PT_FLOW_WGTS[@]}"; do
      for pt_spill_wgt in "${PT_SPILL_WGTS[@]}"; do
        for action_cost in "${ACTION_COSTS[@]}"; do
          for jug_resting_wgt in "${JUG_RESTING_WGTS[@]}"; do
            for jug_velocity_wgt in "${JUG_VELOCITY_WGTS[@]}"; do
              for signal_noise in "${SIGNAL_NOISE[@]}"; do
                # Run the Python script with current combination
                $PYTHON_CMD \
                  --env-id $ENV_ID \
                  --seed $SEED \
                  $SAVE_MODEL \
                  --buffer_size $BUFFER_SIZE \
                  --total_timesteps $TOTAL_TIMESTEPS \
                  $CAPTURE_VIDEO \
                  --output-dir $OUTPUT_DIR \
                  --target_level_wgt $target_level_wgt \
                  --pt_cup_wgt $pt_cup_wgt \
                  --pt_flow_wgt $pt_flow_wgt \
                  --pt_spill_wgt $pt_spill_wgt \
                  --action_cost $action_cost \
                  --jug_resting_wgt $jug_resting_wgt \
                  --jug_velocity_wgt $jug_velocity_wgt \
                  --distance_wgt $DISTANCE_WGT \
                  --exploration_noise $EXPLORATION_NOISE \
                  --signal_noise $signal_noise \
                  --min_signal_noise $MIN_SIGNAL_NOISE \
                  --max_signal_noise $MAX_SIGNAL_NOISE \
                  --time_penalty $TIME_PENALTY
              done
            done
          done
        done
      done
    done
  done
done