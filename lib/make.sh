#!/usr/bin/env bash

CUDA_PATH=/usr/local/cuda/

python setup.py build_ext --inplace
rm -rf build

CUDA_ARCH="-gencode arch=compute_30,code=sm_30 \
           -gencode arch=compute_35,code=sm_35 \
           -gencode arch=compute_37,code=sm_37 \
           -gencode arch=compute_50,code=sm_50 \
           -gencode arch=compute_52,code=sm_52"

# compile roi_pooling
cd roi_pooling/src
echo "Compiling roi pooling kernels by nvcc..."
nvcc -c -o roi_pooling.cu.o roi_pooling_kernel.cu \
	 -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC $CUDA_ARCH
cd ../
python build.py


# compile roi_align
cd ../
cd roi_align/src/cuda
echo "Compiling roi align kernels by nvcc..."
nvcc -c -o crop_and_resize_kernel.cu.o crop_and_resize_kernel.cu -x cu -Xcompiler -fPIC $CUDA_ARCH

# nvcc -c -o ./cuda/crop_and_resize_kernel.cu.o ./cuda/crop_and_resize_kernel.cu -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC $CUDA_ARCH
cd ../../
python build.py