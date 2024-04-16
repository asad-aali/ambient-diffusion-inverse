EXPERIMENT_NAME=ffhq_supervised_noisy
GPUS_PER_NODE=1
MODEL_PATH=/home/asad/ambient-diffusion/models/checkpoints/ffhq
MEAS_PATH=/home/asad/ambient-diffusion/data/ffhq-64x64.zip
GPU=3

for corr in 4 6 8
do
    torchrun --standalone --nproc_per_node=$GPUS_PER_NODE \
        solve_inverse_problems_noisy.py \
        --corruption_probability=0 --delta_probability=0 \
        --operator_corruption_probability=$corr \
        --experiment_name=$EXPERIMENT_NAME-op_corr$corr \
        --outdir=results/$EXPERIMENT_NAME-op_corr$corr \
        --network=$MODEL_PATH \
        --measurements_path=$MEAS_PATH \
        --ref=$MODEL_PATH \
        --training_options_loc=$MODEL_PATH/training_options.json \
        --gpu=$GPU \
        --corruption_pattern=averaging \
        --downsampling_factor=$corr \
        --mask_full_rgb=True \
        --with_wandb=False \
        --steps=300 \
        --num=36 \
        --batch=36
done