#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 14
#SBATCH --mem 64G
#SBATCH --time 2-1
echo STARTING AT `date`

cd
rsync -a ../../scratch/laechler/NTDS/* $TMPDIR/

module load gcc
module load python
echo Successfully loaded module environment

cd $TMPDIR/

virtualenv --system-site-packages venvs/pytorch
source venvs/pytorch/bin/activate

pip3 install torch===1.3.1 torchvision===0.4.2 -f https://download.pytorch.org/whl/torch_stable.html

echo Start running python code at `date`
python trainNeuralNet.py >> ../../scratch/laechler/NTDS/nn_training.log
echo Finished running python code at `date`

echo Start copying stuff at `date`
rsync -a *.pt ../../scratch/laechler/NTDS/
rsync -a *.txt ../../scratch/laechler/NTDS/
echo Finished copying final stuff into scratch at `date`

exit
