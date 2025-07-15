#!/bin/bash
#SBATCH --job-name c3dgs                  # 任务名叫 example
#SBATCH --array 0-1                        # 提交 100 个子任务，序号分别为 0,1,2,...99
#SBATCH --gres gpu:a100:1                   # 每个子任务都用一张 A100 GPU
#SBATCH --time 0-4:00:00                    # 子任务 1 天 1 小时就能跑完
#SBATCH --output %A_%a.out                  # 100个程序输出重定向到 [任务id]_[子任务序号].out
#SBATCH --error %A_%a.err                   # 错误信息重定向到 [任务id]_[子任务序号].err

# 打印一些调试信息
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Running on host: $(hostname)"
echo "Starting at: $(date)"

# 根据任务ID激活不同的Conda环境并执行不同的脚本
case ${SLURM_ARRAY_TASK_ID} in
  0)
    for lmbda in 0.004 0.003 0.002 0.001 0.0005; do
        for scene in 'playroom'; do
            mask_lr_final = 0.00008 * lmbda / 0.001
            python train.py -s data/blending/${scene} --eval --lod 0 --voxel_size 0.005 --update_init_factor 16 --iterations 30000 -m outputs/blending/${scene}/${lmbda} --lmbda ${lmbda} --mask_lr_final ${mask_lr_final}

        done
    done
    ;;
  1)
    for lmbda in 0.004 0.003 0.002 0.001 0.0005; do
        for scene in 'drjohnson'; do
            mask_lr_final = 0.00008 * lmbda / 0.001
            python train.py -s data/blending/${scene} --eval --lod 0 --voxel_size 0.005 --update_init_factor 16 --iterations 30000 -m outputs/blending/${scene}/${lmbda} --lmbda ${lmbda} --mask_lr_final ${mask_lr_final}

        done
    done
    ;;
  # 添加更多的任务ID、对应的Conda环境和脚本
  *)
    echo "No script and environment defined for task ID ${SLURM_ARRAY_TASK_ID}"
    ;;
esac

# 结束时间
echo "Finished at: $(date)"