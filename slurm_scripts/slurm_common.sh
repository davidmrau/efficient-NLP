USER=`accinfo | grep NRC | awk '{print $3}'`
#Loading modules
module purge
module load pre2019
module load 2019
module load eb
module load Python/3.6.3-foss-2017b
module load cuDNN/7.0.5-CUDA-9.0.176
module load NCCL/2.0.5-CUDA-9.0.176 

cd ..

FILE_NAME=$1
export FILE_NAME

bash telegram.sh "${USER} ${FILE_NAME} ${SLURM_JOBID} Started"
source slurm_scripts/${1}
bash telegram.sh "${USER} ${FILE_NAME} ${SLURM_JOBID} Finished"


