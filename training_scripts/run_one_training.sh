#!/bin/bash

# Fixed parameters
PYTHON_CMD="python /home/carola/masterthesis/cleanrl/cleanrl/td3_continuous_action_jax.py"
ENV_ID="PouringEnvIsaacVisualNoGaze-v0"
SEED=42
SAVE_MODEL="--save-model"
BUFFER_SIZE=50000
TOTAL_TIMESTEPS=500000
CAPTURE_VIDEO="--capture_video"
OUTPUT_DIR="/home/carola/masterthesis/cleanrl/cleanrl/outputs/"

# Custom parameters (set them here before running)
TARGET_LEVEL_WGT=0.5
PT_CUP_WGT=15.0
PT_FLOW_WGT=0 
PT_SPILL_WGT=-30.0
ACTION_COST=-0.01
JUG_RESTING_WGT=-0.0000001
JUG_VELOCITY_WGT=0
DISTANCE_WGT=0
FOVEA_RADIUS=50

# Run the Python script with chosen parameters
$PYTHON_CMD \
  --env-id $ENV_ID \
  --seed $SEED \
  $SAVE_MODEL \
  --buffer_size $BUFFER_SIZE \
  --total_timesteps $TOTAL_TIMESTEPS \
  $CAPTURE_VIDEO \
  --output-dir $OUTPUT_DIR \
  --target_level_wgt $TARGET_LEVEL_WGT \
  --pt_cup_wgt $PT_CUP_WGT \
  --pt_flow_wgt $PT_FLOW_WGT \
  --pt_spill_wgt $PT_SPILL_WGT \
  --action_cost $ACTION_COST \
  --jug_resting_wgt $JUG_RESTING_WGT \
  --jug_velocity_wgt $JUG_VELOCITY_WGT\
  --distance_wgt $DISTANCE_WGT \
  --fovea_radius $FOVEA_RADIUS
