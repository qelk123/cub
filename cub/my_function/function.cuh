struct MyStruct
{
  template <typename NE_list_Iter,
            typename Index_list_Iter,
            typename NV_list_Iter,
            typename Bias_T,
            typename ValueT>
  static __device__ __forceinline__ void compute_before_scatter(
          int             ITEMS_PER_THREAD,
          int             BLOCK_THREADS,
          int             compute_item,
          int             ne_nnz,
          int             nv_nnz,
          Bias_T          tile_bias,
          int*            batch_list,
          int             batch_list_size,
          NE_list_Iter        ne_list,//input ne list
          Index_list_Iter     ne_index,//sparse matrix index list
          NV_list_Iter        nv_list,//input nv list
          ValueT*         input_shared,
          ValueT*         output_shared)
          {
            #pragma unroll
            for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
            {
            
                int idx_block = threadIdx.x + (ITEM * BLOCK_THREADS);
                int idx_global = idx_block + tile_bias.y;

                __syncthreads();

                for (int i = 0 ; i < batch_list[0]; i++ ) {
                  if (threadIdx.x + i * BLOCK_THREADS < (compute_item - (ITEM * BLOCK_THREADS)) * batch_list[0]) {
                    input_shared[threadIdx.x + i * BLOCK_THREADS] = ne_list[((tile_bias.y + (ITEM * BLOCK_THREADS)) * batch_list[0]) + threadIdx.x + i * BLOCK_THREADS];
                    // tempstorage.aliasable.batch_op.input_buffer_0[threadIdx.x * batch_list[0] + i] = ne_list[((tile_bias.y +  (ITEM * BLOCK_THREADS)) * batch_list[0]) + threadIdx.x * batch_list[0] + i];
                  }
                }
                __syncthreads();


                if (idx_block < compute_item)
                {
                  //parts that need to be generated automatically
                  //实际在代码生成的过程中，index的extent是直接写死的，而不是通过参数读入
                  //----------------begin-----------------
                    // ValueT  tmp_0[3];
                    // //step 1
                    // //spacial iter
                    // for (int batch_idx_0=0; batch_idx_0 < batch_list[0]; ++batch_idx_0) {
                    //   //init local memory
                    //   tmp_0[batch_idx_0] = 0;
                    //   //reduce iter
                    //   for (int batch_idx_1=0; batch_idx_1 < batch_list[1]; ++batch_idx_1) {
                    //     //load ne:global buffer + global index 
                    //     ValueT  value                   = ne_list[0][((batch_idx_0 * batch_list[1] + batch_idx_1)  * ne_nnz) + idx_global];
                    //     //load nv:global buffer + global dense index + column index buffer + global index
                    //     ValueT  vector_value            = nv_list[0][(batch_idx_0 * batch_list[1] + batch_idx_1) * nv_nnz + ne_index[0][idx_global]];
                    //     //compute:store to local memory (register)
                    //     tmp_0[batch_idx_0]             += value * vector_value;
                    //   }
                    // }
                    // //step 2
                    // for (int batch_idx_0=0; batch_idx_0 < batch_list[0]; ++batch_idx_0) {
                    //   output_shared[idx_block * batch_list[0] + batch_idx_0] = tmp_0[batch_idx_0];
                    // }







                    for (int batch_idx_0=0; batch_idx_0 < batch_list[0]; ++batch_idx_0) {
                      #ifdef USE_LIST
                      ValueT  value                   = ne_list[0][(batch_idx_0 * ne_nnz) + idx_global];
                      ValueT  vector_value            = nv_list[0][batch_idx_0 * nv_nnz + ne_index[0][idx_global]];
                      #else
                      ValueT  value                         = input_shared[threadIdx.x * batch_list[0] + batch_idx_0];
                      // ValueT  value                         = ne_list[idx_global * batch_list[0] + batch_idx_0];
                      ValueT  vector_value            = nv_list[batch_idx_0 * nv_nnz + ne_index[idx_global]];
                      #endif
                      output_shared[idx_block * batch_list[0] + batch_idx_0] = value * vector_value;
                    }

                  //----------------end-----------------
                }
            }
          }
};
