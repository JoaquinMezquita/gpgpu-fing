#!/bin/bash
#SBATCH --job-name=mitrabajo
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=00:05:00

#SBATCH --gres=gpu:1

#SBATCH --partition=cursos
#SBATCH --qos=gpgpu

PATH=$PATH:/usr/local/cuda/bin
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

nvcc "$1.cu" -o "$1"
nvprof -o salida --metrics achieved_occupancy,gld_throughput,gst_throughput,gld_efficiency,gst_efficiency,sm_efficiency,warp_execution_efficiency -f ./$1
nvprof -i salida
rm -f report1.nsys-rep salida report1.sqlite $1