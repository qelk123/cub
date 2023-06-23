const int scatter_op_num = 2;
const int batch_size_list[scatter_op_num] = { 3, 3};
ValueT* result_addr_list[scatter_op_num] = { easier_params.d_vector_y_0, easier_params.d_vector_y_1};