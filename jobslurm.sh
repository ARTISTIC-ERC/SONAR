#!/bin/sh

#SBATCH --job-name=python_trial
#SBATCH --partition=long               # submission queue (normal or bigmem or bigpu or quadgpu)
#SBATCH --time=3-0:00:00                 # 1-1 means one day and one hour
#SBATCH --mail-type=ALL                  # Type can be BEGIN, END, FAIL, ALL(any statchange).
##SBATCH --mail-user=jia.yu@u-picardie.fr # e-mail notification
#SBATCH --output=job_seq-%j.out          # if --error is absent, includes also the errors
#SBATCH --mem=10G                         # T-tera, G-giga, M-mega
#SBATCH --nodes=1                        # 4 cpus CPU Numbers
#SBATCH --ntasks-per-node=24
##SBATCH --cpus-per-task=4

echo "-----------------------------------------------------------"
echo "hostname                     =   $(hostname)"
echo "SLURM_JOB_NAME               =   $SLURM_JOB_NAME"
echo "SLURM_SUBMIT_DIR             =   $SLURM_SUBMIT_DIR"
echo "SLURM_JOBID                  =   $SLURM_JOBID"
echo "SLURM_JOB_ID                 =   $SLURM_JOB_ID"
echo "SLURM_NODELIST               =   $SLURM_NODELIST"
echo "SLURM_JOB_NODELIST           =   $SLURM_JOB_NODELIST"
echo "SLURM_TASKS_PER_NODE         =   $SLURM_TASKS_PER_NODE"
echo "SLURM_JOB_CPUS_PER_NODE      =   $SLURM_JOB_CPUS_PER_NODE"
echo "SLURM_TOPOLOGY_ADDR_PATTERN  =   $SLURM_TOPOLOGY_ADDR_PATTERN"module 
echo "SLURM_TOPOLOGY_ADDR          =   $SLURM_TOPOLOGY_ADDR"
echo "SLURM_CPUS_ON_NODE           =   $SLURM_CPUS_ON_NODE"
echo "SLURM_NNODES                 =   $SLURM_NNODES"
echo "SLURM_JOB_NUM_NODES          =   $SLURM_JOB_NUM_NODES"
echo "SLURMD_NODENAME              =   $SLURMD_NODENAME"
echo "SLURM_NTASKS                 =   $SLURM_NTASKS"
echo "SLURM_NPROCS                 =   $SLURM_NPROCS"
echo "SLURM_MEM_PER_NODE           =   $SLURM_MEM_PER_NODE"
echo "SLURM_PRIO_PROCESS           =   $SLURM_PRIO_PROCESS"
echo "-----------------------------------------------------------"

# USER Commands

workfile=`pwd`

mkdir /scratch/jyu/$SLURM_JOB_ID

# Move to /scratch or launch from id

cp particles_CC.py /scratch/jyu/$SLURM_JOB_ID
cp user_input_SONAR.ini /scratch/jyu/$SLURM_JOB_ID
cp simulate_SONAR_RFB.py /scratch/jyu/$SLURM_JOB_ID
cp initialize_kMC_run.py /scratch/jyu/$SLURM_JOB_ID

cd /scratch/jyu/$SLURM_JOB_ID

# special commands for openmpi/intel

module load python36
module load py3/matplotlib
module load py3/scikit-learn
module load py3/pandas

pip3 install --upgrade pip
pip3 install --upgrade --user pandas

pip3 install --upgrade --user tensorflow-gpu

python3.6 simulate_SONAR_RFB.py

mv * $workfile

#rmdir ${scratchfile}




