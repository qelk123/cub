// // OffsetT*    s_tile_row_end_offsets_0  = &temp_storage.aliasable.batch_op.merge_items[0].row_end_offset;//前面的tile_num_rows + ITEMS_PER_THREAD项的对应的空间用于保存reduction?
// OffsetT*    s_tile_row_end_offsets_0  = &temp_storage.batch_op.merge_items[0].row_end_offset;//前面的tile_num_rows + ITEMS_PER_THREAD项的对应的空间用于保存reduction?
// OffsetT*    s_tile_row_end_offsets_1  = &temp_storage.batch_op.merge_items2[0].row_end_offset;//前面的tile_num_rows + ITEMS_PER_THREAD项的对应的空间用于保存reduction?
// // OffsetT*    s_tile_row_end_offsets_1;//前面的tile_num_rows + ITEMS_PER_THREAD项的对应的空间用于保存reduction?
// // ValueT*     s_tile_nonzeros_0         = &temp_storage.aliasable.batch_op.merge_items[(tile_num_rows + ITEMS_PER_THREAD)*BATCH_SIZE].nonzero;//先计算乘法并保存到shared memory当中
// ValueT*     s_tile_nonzeros_0         = &temp_storage.batch_op.merge_items[(tile_num_rows + ITEMS_PER_THREAD)*BATCH_SIZE].nonzero;//先计算乘法并保存到shared memory当中
// ValueT*     s_tile_nonzeros_1         = &temp_storage.batch_op.merge_items2[(tile_num_rows + ITEMS_PER_THREAD)*BATCH_SIZE].nonzero;//先计算乘法并保存到shared memory当中
// // ValueT*     s_tile_nonzeros_1;//先计算乘法并保存到shared memory当中

// const int SCATTER_OP_NUM = 2;
// const int BATCH_SIZE_LIST[SCATTER_OP_NUM] = {BATCH_SIZE, BATCH_SIZE};
// // const int MAX_BATCH_SIZE = BATCH_SIZE;
// ValueT* s_tile_nonzeros_list[SCATTER_OP_NUM] = {s_tile_nonzeros_0, s_tile_nonzeros_1};
// OffsetT* s_tile_row_end_offsets_list[SCATTER_OP_NUM] = {s_tile_row_end_offsets_0, s_tile_row_end_offsets_1};

// ValueT* global_result_addr_list[SCATTER_OP_NUM] = {easier_params.d_vector_y, easier_params.d_vector_y_2};


// ValueT* e1_shared = temp_storage.batch_op.batch_aliasable.s_slot_0;
// OffsetT* s_tile_row_end_offsets_B_einsum = &temp_storage.batch_op.merge_items_B_einsum[0].row_end_offset;
// ValueT* s_tile_nonzeros_B_einsum = &temp_storage.batch_op.merge_items_B_einsum[ (tile_num_rows + ITEMS_PER_THREAD) * 4].nonzero;
// OffsetT* s_tile_row_end_offsets_B_einsum_1 = &temp_storage.batch_op.merge_items_B_einsum_1[0].row_end_offset;
// ValueT* s_tile_nonzeros_B_einsum_1 = &temp_storage.batch_op.merge_items_B_einsum_1[ (tile_num_rows + ITEMS_PER_THREAD) * 4].nonzero;
// const int SCATTER_OP_NUM = 2;
// const int BATCH_SIZE_LIST[SCATTER_OP_NUM] = { 4, 4};
// ValueT* s_tile_nonzeros_list[SCATTER_OP_NUM] = { s_tile_nonzeros_B_einsum, s_tile_nonzeros_B_einsum_1};
// OffsetT* s_tile_row_end_offsets_list[SCATTER_OP_NUM] = { s_tile_row_end_offsets_B_einsum, s_tile_row_end_offsets_B_einsum_1};
// ValueT* global_result_addr_list[SCATTER_OP_NUM] = { easier_params.d_vector_y_0, easier_params.d_vector_y_1};

ValueT* e1_shared = temp_storage.batch_op.batch_aliasable.s_slot_0;
OffsetT* s_tile_row_end_offsets_B_einsum = &temp_storage.batch_op.merge_items_B_einsum[0].row_end_offset;
ValueT* s_tile_nonzeros_B_einsum = &temp_storage.batch_op.merge_items_B_einsum[ (tile_num_rows + ITEMS_PER_THREAD) * 3].nonzero;
OffsetT* s_tile_row_end_offsets_B_einsum_1 = &temp_storage.batch_op.merge_items_B_einsum_1[0].row_end_offset;
ValueT* s_tile_nonzeros_B_einsum_1 = &temp_storage.batch_op.merge_items_B_einsum_1[ (tile_num_rows + ITEMS_PER_THREAD) * 3].nonzero;
const int SCATTER_OP_NUM = 2;
const int BATCH_SIZE_LIST[SCATTER_OP_NUM] = { 3, 3};
ValueT* s_tile_nonzeros_list[SCATTER_OP_NUM] = { s_tile_nonzeros_B_einsum, s_tile_nonzeros_B_einsum_1};
OffsetT* s_tile_row_end_offsets_list[SCATTER_OP_NUM] = { s_tile_row_end_offsets_B_einsum, s_tile_row_end_offsets_B_einsum_1};
ValueT* global_result_addr_list[SCATTER_OP_NUM] = { easier_params.d_vector_y_0, easier_params.d_vector_y_1};