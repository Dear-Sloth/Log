#!/bin/bash

#SBATCH -p a100             
#SBATCH -J LLM-sft_multi
#SBATCH -N 3                      # 64x8x4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4            
#SBATCH --mail-type=all
#SBATCH --mail-user=@qq.com
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --overcommit             # needed for pytorch
#SBATCH --output=out.log
#SBATCH --exclusive
# should be modified to train_sft_llama.sh, train_rm_llama.sh, train_ppo_llama, etc.
readonly training_script="train_sft_llama_test.sh" 
readonly GPUS_PER_NODE=4

readonly PROJECT_PATH=$(cd /dssg/home/acct-msejzh/msejzh-user1/OpenRLHF/examples/scripts; pwd)
readonly JOBLOG="$(pwd)/logs/$training_script-$SLURM_JOB_ID.log"
mkdir logs

# Job start
echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} started ..." &>> ${JOBLOG}

# load training commands
source ./${training_script} slurm
echo training_commands &>> ${JOBLOG}
echo $training_commands &>> ${JOBLOG}

# master addr and port
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9901

module load miniconda3 gcc/11.2.0 
module load cuda/11.8.0
source activate openrlhf


srun bash -c "cd ${PROJECT_PATH}; chmod +x ./build_openrlhf.sh; ./build_openrlhf.sh; torchrun \
--nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank \$SLURM_PROCID \
--master_addr $MASTER_ADDR --master_port $MASTER_PORT ${training_commands}" &>> ${JOBLOG}

echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} stopped ..." &>> ${JOBLOG}