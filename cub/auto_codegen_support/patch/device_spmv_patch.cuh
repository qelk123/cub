
/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/**
 * \file
 * cub::DeviceSpmv provides device-wide parallel operations for performing sparse-matrix * vector multiplication (SpMV).
 */

#pragma once

#include <stdio.h>
#include <iterator>
#include <limits>

#include <cub/config.cuh>
// #include <cub/device/dispatch/dispatch_spmv_orig.cuh>
#include "./dispatch_spmv_orig_patch.cuh"
#include <cub/util_deprecated.cuh>

CUB_NAMESPACE_BEGIN


/**
 * \brief DeviceSpmv provides device-wide parallel operations for performing sparse-matrix * dense-vector multiplication (SpMV).
 * \ingroup SingleModule
 *
 * \par Overview
 * The [<em>SpMV computation</em>](http://en.wikipedia.org/wiki/Sparse_matrix-vector_multiplication)
 * performs the matrix-vector operation
 * <em>y</em> = <b>A</b>*<em>x</em> + <em>y</em>,
 * where:
 *  - <b>A</b> is an <em>m</em>x<em>n</em> sparse matrix whose non-zero structure is specified in
 *    [<em>compressed-storage-row (CSR) format</em>](http://en.wikipedia.org/wiki/Sparse_matrix#Compressed_row_Storage_.28CRS_or_CSR.29)
 *    (i.e., three arrays: <em>values</em>, <em>row_offsets</em>, and <em>column_indices</em>)
 *  - <em>x</em> and <em>y</em> are dense vectors
 *
 * \par Usage Considerations
 * \cdp_class{DeviceSpmv}
 *
 */
struct Easier_Struct
{
    /******************************************************************//**
     * \name CSR matrix operations
     *********************************************************************/
    //@{

    /**
     * \brief This function performs the matrix-vector operation <em>y</em> = <b>A</b>*<em>x</em>.
     *
     * \par Snippet
     * The code snippet below illustrates SpMV upon a 9x9 CSR matrix <b>A</b>
     * representing a 3x3 lattice (24 non-zeros).
     *
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/device/device_spmv.cuh>
     *
     * // Declare, allocate, and initialize device-accessible pointers for input matrix A, input vector x,
     * // and output vector y
     * int    num_rows = 9;
     * int    num_cols = 9;
     * int    num_nonzeros = 24;
     *
     * float* d_values;  // e.g., [1, 1, 1, 1, 1, 1, 1, 1,
     *                   //        1, 1, 1, 1, 1, 1, 1, 1,
     *                   //        1, 1, 1, 1, 1, 1, 1, 1]
     *
     * int*   d_column_indices; // e.g., [1, 3, 0, 2, 4, 1, 5, 0,
     *                          //        4, 6, 1, 3, 5, 7, 2, 4,
     *                          //        8, 3, 7, 4, 6, 8, 5, 7]
     *
     * int*   d_row_offsets;    // e.g., [0, 2, 5, 7, 10, 14, 17, 19, 22, 24]
     *
     * float* d_vector_x;       // e.g., [1, 1, 1, 1, 1, 1, 1, 1, 1]
     * float* d_vector_y;       // e.g., [ ,  ,  ,  ,  ,  ,  ,  ,  ]
     * ...
     *
     * // Determine temporary device storage requirements
     * void*    d_temp_storage = NULL;
     * size_t   temp_storage_bytes = 0;
     * cub::DeviceSpmv::CsrMV(d_temp_storage, temp_storage_bytes, d_values,
     *     d_row_offsets, d_column_indices, d_vector_x, d_vector_y,
     *     num_rows, num_cols, num_nonzeros);
     *
     * // Allocate temporary storage
     * cudaMalloc(&d_temp_storage, temp_storage_bytes);
     *
     * // Run SpMV
     * cub::DeviceSpmv::CsrMV(d_temp_storage, temp_storage_bytes, d_values,
     *     d_row_offsets, d_column_indices, d_vector_x, d_vector_y,
     *     num_rows, num_cols, num_nonzeros);
     *
     * // d_vector_y <-- [2, 3, 2, 3, 4, 3, 2, 3, 2]
     *
     * \endcode
     *
     * \tparam ValueT       <b>[inferred]</b> Matrix and vector value type (e.g., /p float, /p double, etc.)
     */

    template <int BATCH_SIZE,typename ValueT>
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
    CUB_RUNTIME_FUNCTION static cudaError_t Easier(void *d_temp_storage,
                                                  size_t &temp_storage_bytes,
                                                  const ValueT *d_values,
                                                  const int *d_row_offsets,
                                                  const int *d_column_indices,
                                                  const ValueT *d_vector_x,
                                                  ValueT *d_vector_y,
                                                  int num_rows,
                                                  int num_cols,
                                                  int num_nonzeros,
                                                  cudaStream_t stream,
                                                  bool debug_synchronous)
    {
      CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

        EasierParams<ValueT, int> spmv_params;
        spmv_params.d_values             = d_values;
        spmv_params.d_row_end_offsets    = d_row_offsets + 1;
        spmv_params.d_column_indices     = d_column_indices;
        spmv_params.d_vector_x           = d_vector_x;
        spmv_params.d_vector_y           = d_vector_y;
        spmv_params.num_rows             = num_rows;
        spmv_params.num_cols             = num_cols;
        spmv_params.num_nonzeros         = num_nonzeros;
        spmv_params.alpha                = ValueT{1};
        spmv_params.beta                 = ValueT{0};

        return DispatchEasier<ValueT, int, BATCH_SIZE>::Dispatch(
            d_temp_storage,
            temp_storage_bytes,
            spmv_params,
            stream);
    }


    template <int BATCH_SIZE,typename ValueT,typename OffsetT>
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
    CUB_RUNTIME_FUNCTION static cudaError_t Easier(void *d_temp_storage,
                                                  size_t &temp_storage_bytes,
                                                  EasierParams<ValueT, OffsetT>  params,
                                                  cudaStream_t stream,
                                                  bool debug_synchronous)
    {
      CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG
        return DispatchEasier<ValueT, int, BATCH_SIZE>::Dispatch(
            d_temp_storage,
            temp_storage_bytes,
            params,
            stream);
    }

};



CUB_NAMESPACE_END


