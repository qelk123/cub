#!/bin/bash


# kernel=(933)
# block_size_array=(64)
# tile_item_array=(5)
# batch_dim_array=(2)
# kernel=(933 2388)
kernel=(2388)
block_size_array=(32 64 128 256 512)
tile_item_array=(1 2 3 4 5 7 9)
# batch_dim_array=(1 2 3 4 8 9 16 25 27 32)
batch_dim_array=(4)
echo "MATRIX BATCH_SIZE BLOCK_SIZE ITEM_PER_THREAD time"

for kernel in ${kernel[@]}
do
  for batch_dim in ${batch_dim_array[@]}
  do
    for tile_item in ${tile_item_array[@]}
    do
      for block_size in ${block_size_array[@]}
      do
        # /usr/local/cuda-11.3/bin/nvcc -D BLOCK_SIZE=${3} -D ITEM_PER_THREAD=${4} -D BATCH_SIZE=${5} --resource-usage -O3 -gencode=arch=compute_80,code=\"sm_80,compute_80\" -lineinfo -o ${1} ${1}.cu -diag-suppress 2464 -Xptxas -v -Xcudafe -#  -Xcompiler -ffloat-store -I/home/v-yinuoliu/yinuoliu/code/thrust/examples -I/home/v-yinuoliu/yinuoliu/code/thrust -I/home/v-yinuoliu/yinuoliu/code/thrust//dependencies/libcudacxx/include -I/home/v-yinuoliu/yinuoliu/code/thrust//dependencies/cub/ -I./ -lcusparse
        /usr/local/cuda-11.3/bin/nvcc -D BLOCK_SIZE=${block_size} -D ITEM_PER_THREAD=${tile_item} -D BATCH_SIZE_D=${batch_dim} -O3 --disable-warnings -gencode=arch=compute_80,code=\"sm_80,compute_80\" -lineinfo -o gpu_spmv ./gpu_spmv.cu  -I/home/v-yinuoliu/yinuoliu/code/thrust/examples -I/home/v-yinuoliu/yinuoliu/code/thrust -I/home/v-yinuoliu/yinuoliu/code/thrust//dependencies/libcudacxx/include -I/home/v-yinuoliu/yinuoliu/code/thrust//dependencies/cub/ -I./ -lcusparse
        if [ $? -eq 0 ]; then
          # ./gpu_spmv --quiet --mtx=/home/v-yinuoliu/yinuoliu/code/SparseCodegen/matrix${kernel}.mtx --i=1
          ./gpu_spmv --quiet --mtx=${kernel} --i=100
        else
          echo "compile failed!"
        fi
      done
    done
  done
done