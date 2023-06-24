struct MyStruct
{
  template <typename NE_list_Iter,typename Index_list_Iter,typename NV_list_Iter,typename Bias_T,typename ValueT>
  static __device__ __forceinline__ void compute_before_scatter_auto_gen( int item_pt,int block_size,int compute_item,Bias_T tile_bias,
  NV_list_Iter v1,
  NE_list_Iter e1,
  ValueT* e1_shared,
  Index_list_Iter G_0,
  ValueT* B_einsum_shared,
  ValueT* B_einsum_1_shared) {
  #pragma unroll
  for (int ITEM = 0; ITEM < item_pt; ++ITEM)
  {

  int idx_block = threadIdx.x + (ITEM * block_size);
  int idx_global = idx_block + tile_bias.y;
  if (idx_block < compute_item) {

  ValueT B_gather_src[ 3*1] = {0};
  for(int axis_8 = 0; axis_8 < 3; axis_8++ ) {
  B_gather_src[( ( 0 *3 + axis_8) * 1 )]=(v1[( (  0 * 3 + axis_8) * 21036+ G_0[idx_global] )]);
  }
  for(int axis_5 = 0; axis_5 < 3; axis_5++ ) {
  for(int axis_8 = 0; axis_8 < 3; axis_8++ ) {
  e1_shared[(( 0  * 3 + axis_5) * 3 + axis_8) * block_size + threadIdx.x]=e1[(( 0  * 3 + axis_5) * 3 + axis_8) * 303468 + idx_global];
  }
  }
  ValueT B_einsum[ 3*1] = {0};
  for(int axis_5 = 0; axis_5 < 3; axis_5++ ) {
  for(int axis_8 = 0; axis_8 < 3; axis_8++ ) {
  B_einsum[( ( 0 *3 + axis_5) * 1 )]+=(e1_shared[ ( ( (  0 * 3 + axis_5) * 3 + axis_8) *  block_size + threadIdx.x )]* B_gather_src[ ( (  0 * 3 + axis_8) * 1 )]);
  }
  }
  for(int axis_5 = 0; axis_5 < 3; axis_5++ ) {
  B_einsum_shared[( ( 0 *3 + axis_5) *  compute_item + idx_block)]=B_einsum[( ( 0 *3 + axis_5) * 1 )];
  }

  ValueT B_einsum_1[ 3*1] = {0};
  for(int axis_2 = 0; axis_2 < 3; axis_2++ ) {
  for(int axis_7 = 0; axis_7 < 3; axis_7++ ) {
  B_einsum_1[( ( 0 *3 + axis_2) * 1 )]+=(e1_shared[ ( ( (  0 * 3 + axis_2) * 3 + axis_7) *  block_size + threadIdx.x )]* B_gather_src[ ( (  0 * 3 + axis_7) * 1 )]);
  }
  }
  for(int axis_2 = 0; axis_2 < 3; axis_2++ ) {
  B_einsum_1_shared[( ( 0 *3 + axis_2) *  compute_item + idx_block)]=B_einsum_1[( ( 0 *3 + axis_2) * 1 )];
  }
  }
  }
  }
};



