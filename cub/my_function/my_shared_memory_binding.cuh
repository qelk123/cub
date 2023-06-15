// OffsetT*    s_tile_row_end_offsets_0  = &temp_storage.aliasable.batch_op.merge_items[0].row_end_offset;//前面的tile_num_rows + ITEMS_PER_THREAD项的对应的空间用于保存reduction?
OffsetT*    s_tile_row_end_offsets_0  = &temp_storage.batch_op.merge_items[0].row_end_offset;//前面的tile_num_rows + ITEMS_PER_THREAD项的对应的空间用于保存reduction?
OffsetT*    s_tile_row_end_offsets_1  = &temp_storage.batch_op.merge_items2[0].row_end_offset;//前面的tile_num_rows + ITEMS_PER_THREAD项的对应的空间用于保存reduction?
// OffsetT*    s_tile_row_end_offsets_1;//前面的tile_num_rows + ITEMS_PER_THREAD项的对应的空间用于保存reduction?
// ValueT*     s_tile_nonzeros_0         = &temp_storage.aliasable.batch_op.merge_items[(tile_num_rows + ITEMS_PER_THREAD)*BATCH_SIZE].nonzero;//先计算乘法并保存到shared memory当中
ValueT*     s_tile_nonzeros_0         = &temp_storage.batch_op.merge_items[(tile_num_rows + ITEMS_PER_THREAD)*BATCH_SIZE].nonzero;//先计算乘法并保存到shared memory当中
ValueT*     s_tile_nonzeros_1         = &temp_storage.batch_op.merge_items2[(tile_num_rows + ITEMS_PER_THREAD)*BATCH_SIZE].nonzero;//先计算乘法并保存到shared memory当中
// ValueT*     s_tile_nonzeros_1;//先计算乘法并保存到shared memory当中

const int SCATTER_OP_NUM = 2;
const int BATCH_SIZE_LIST[2] = {BATCH_SIZE, BATCH_SIZE};
const int MAX_BATCH_SIZE = BATCH_SIZE;
ValueT* s_tile_nonzeros_list[2] = {s_tile_nonzeros_0, s_tile_nonzeros_1};
OffsetT* s_tile_row_end_offsets_list[2] = {s_tile_row_end_offsets_0, s_tile_row_end_offsets_1};