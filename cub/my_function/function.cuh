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





                    ValueT tmp_reg[BATCH_SIZE] = {0};
                    int col_index = ne_index[idx_global];
                    for (int batch_idx_0=0; batch_idx_0 < batch_list[0]; ++batch_idx_0) {
                      for (int batch_idx_1=0; batch_idx_1 < batch_list[0]; ++batch_idx_1) {
                        ValueT  vector_value                  = nv_list[batch_idx_1 * nv_nnz + col_index];
                        ValueT  value                         = ne_list[(batch_idx_0 * batch_list[0] + batch_idx_1) * ne_nnz + (idx_global)];
                        tmp_reg[batch_idx_0] += value * vector_value;
                        output_shared[idx_block * batch_list[0] + batch_idx_0] = tmp_reg[batch_idx_0];
                      }
                    }

                  //----------------end-----------------
                }
            }
          }



// template <typename NE_list_Iter,typename Index_list_Iter,typename NV_list_Iter,typename Bias_T,typename ValueT>
// static __device__ __forceinline__ void compute_before_scatter_auto_gen( int item_pt,int block_size,int compute_item,Bias_T tile_bias,
// NV_list_Iter v1,
// NE_list_Iter e1,
// Index_list_Iter G_0,
// ValueT* B_einsum_shared,
// ValueT* B_einsum_shared1
// ) {
// #pragma unroll
// for (int ITEM = 0; ITEM < item_pt; ++ITEM)
// {

// int idx_block = threadIdx.x + (ITEM * block_size);
// int idx_global = idx_block + tile_bias.y;
// if (idx_block < compute_item) {

// ValueT B_einsum[ 4*1] = {0};
// for(int axis_2 = 0; axis_2 < 4; axis_2++ ) {
// for(int axis_4 = 0; axis_4 < 4; axis_4++ ) {
// // B_einsum[( ( 0 *4 + axis_2) * 1 )]+=(e1[ ( ( (  0 * 4 + axis_2) * 4 + axis_4) * 303468 + idx_global )]* (v1[( (  0 * 4 + axis_4) * 21036+ G_0[idx_global] )]));
// B_einsum[( ( 0 *4 + axis_2) * 1 )]+=(e1[ ( ( (  0 * 4 + axis_2) * 4 + axis_4) * 89306020 + idx_global )]* (v1[( (  0 * 4 + axis_4) * 1102824 + G_0[idx_global] )]));
// }
// }
// for(int axis_2 = 0; axis_2 < 4; axis_2++ ) {
// B_einsum_shared[( ( 0 *4 + axis_2) *  compute_item + idx_block)]=B_einsum[( ( 0 *4 + axis_2) * 1 )];
// B_einsum_shared1[( ( 0 *4 + axis_2) *  compute_item + idx_block)]=B_einsum[( ( 0 *4 + axis_2) * 1 )];
// }
// }
// }
// }


// template <typename NE_list_Iter,typename Index_list_Iter,typename NV_list_Iter,typename Bias_T,typename ValueT>
// static __device__ __forceinline__ void compute_before_scatter_auto_gen( int item_pt,int block_size,int compute_item,Bias_T tile_bias,
// NV_list_Iter v1,
// NE_list_Iter e1,
// ValueT* e1_shared,
// Index_list_Iter G_0,
// ValueT* B_einsum_shared,
// ValueT* B_einsum_1_shared) {
// #pragma unroll
// for (int ITEM = 0; ITEM < item_pt; ++ITEM)
// {

// int idx_block = threadIdx.x + (ITEM * block_size);
// int idx_global = idx_block + tile_bias.y;
// if (idx_block < compute_item) {

// ValueT B_gather_src[ 3*1] = {0};
// for(int axis_8 = 0; axis_8 < 3; axis_8++ ) {
// B_gather_src[( ( 0 *3 + axis_8) * 1 )]=(v1[( (  0 * 3 + axis_8) * 21036+ G_0[idx_global] )]);
// }
// for(int axis_5 = 0; axis_5 < 3; axis_5++ ) {
// for(int axis_8 = 0; axis_8 < 3; axis_8++ ) {
// e1_shared[(( 0  * 3 + axis_5) * 3 + axis_8) * block_size + threadIdx.x]=e1[(( 0  * 3 + axis_5) * 3 + axis_8) * 303468 + idx_global];
// }
// }
// ValueT B_einsum[ 3*1] = {0};
// for(int axis_5 = 0; axis_5 < 3; axis_5++ ) {
// for(int axis_8 = 0; axis_8 < 3; axis_8++ ) {
// B_einsum[( ( 0 *3 + axis_5) * 1 )]+=(e1_shared[ ( ( (  0 * 3 + axis_5) * 3 + axis_8) *  block_size + threadIdx.x )]* B_gather_src[ ( (  0 * 3 + axis_8) * 1 )]);
// }
// }
// for(int axis_5 = 0; axis_5 < 3; axis_5++ ) {
// B_einsum_shared[( ( 0 *3 + axis_5) *  compute_item + idx_block)]=B_einsum[( ( 0 *3 + axis_5) * 1 )];
// }

// ValueT B_einsum_1[ 3*1] = {0};
// for(int axis_2 = 0; axis_2 < 3; axis_2++ ) {
// for(int axis_7 = 0; axis_7 < 3; axis_7++ ) {
// B_einsum_1[( ( 0 *3 + axis_2) * 1 )]+=(e1_shared[ ( ( (  0 * 3 + axis_2) * 3 + axis_7) *  block_size + threadIdx.x )]* B_gather_src[ ( (  0 * 3 + axis_7) * 1 )]);
// }
// }
// for(int axis_2 = 0; axis_2 < 3; axis_2++ ) {
// B_einsum_1_shared[( ( 0 *3 + axis_2) *  compute_item + idx_block)]=B_einsum_1[( ( 0 *3 + axis_2) * 1 )];
// }
// }
// }
// }



};



