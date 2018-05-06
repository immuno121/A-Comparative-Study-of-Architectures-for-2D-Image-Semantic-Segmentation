#!/bin/bash
#
#SBATCH --partition=titanx-long    # Partition to submit to <m40-short|m40-long|teslax-short|teslax-long>
#SBATCH --job-name=seq_autoencoder
#SBATCH -o run_logs/seq_autoencoder_res_%j.txt            # output file
#SBATCH -e run_logs/seq_autoencoder_res_%j.err            # File to which STDERR will be written
#SBATCH --ntasks=1
#SBATCH --time=07-00:00:00          # D-HH:MM:SS
#SBATCH --mem=80000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shasvatmukes@cs.umass.edu


module load python2/current
#python temp.py
#source activate py27

python evaluate.py
#python train.py
#python cluster.py
#python seq_exp2.py
#python test.py
#python cvtest.py
#lspci -vnn|grep NVIDIA



sleep 1
exit

