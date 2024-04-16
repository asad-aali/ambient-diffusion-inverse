EXPERIMENT_NAME=ffhq_supervised
GPUS_PER_NODE=1
MODEL_PATH=/home/asad/ambient-diffusion/models/checkpoints/ffhq
MEAS_PATH=/home/asad/ambient-diffusion/data/ffhq-64x64.zip
GPU=0

for corr in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
    torchrun --standalone --nproc_per_node=$GPUS_PER_NODE \
        solve_inverse_problems.py \
        --corruption_probability=0 --delta_probability=0 \
        --operator_corruption_probability=$corr \
        --experiment_name=$EXPERIMENT_NAME-op_corr$corr \
        --outdir=results/$EXPERIMENT_NAME-op_corr$corr \
        --network=$MODEL_PATH \
        --measurements_path=$MEAS_PATH \
        --ref=$MODEL_PATH \
        --training_options_loc=$MODEL_PATH/training_options.json \
        --gpu=$GPU \
        --corruption_pattern=box_masking \
        --downsampling_factor=2 \
        --mask_full_rgb=True \
        --with_wandb=False \
        --steps=300 \
        --num=36 \
        --batch=36
done