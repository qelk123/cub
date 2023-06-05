/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
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
 * cub::AgentSpmv implements a stateful abstraction of CUDA thread blocks for participating in device-wide SpMV.
 */

#pragma once

#include <iterator>

#include "../util_type.cuh"
#include "../block/block_reduce.cuh"
#include "../block/block_scan.cuh"
#include "../block/block_exchange.cuh"
#include "../config.cuh"
#include "../thread/thread_search.cuh"
#include "../thread/thread_operators.cuh"
#include "../iterator/cache_modified_input_iterator.cuh"
#include "../iterator/counting_input_iterator.cuh"
#include "../my_function/function.cuh"

CUB_NAMESPACE_BEGIN


/******************************************************************************
 * Tuning policy
 ******************************************************************************/

/**
 * Parameterizable tuning policy type for AgentSpmv
 */
template <
    int                             _BLOCK_THREADS,                         ///< Threads per thread block
    int                             _ITEMS_PER_THREAD,                      ///< Items per thread (per tile of input)
    int                             _BATCH_SIZE,                            ///< batch size for batch spmv
    CacheLoadModifier               _ROW_OFFSETS_SEARCH_LOAD_MODIFIER,      ///< Cache load modifier for reading CSR row-offsets during search
    CacheLoadModifier               _ROW_OFFSETS_LOAD_MODIFIER,             ///< Cache load modifier for reading CSR row-offsets
    CacheLoadModifier               _COLUMN_INDICES_LOAD_MODIFIER,          ///< Cache load modifier for reading CSR column-indices
    CacheLoadModifier               _VALUES_LOAD_MODIFIER,                  ///< Cache load modifier for reading CSR values
    CacheLoadModifier               _VECTOR_VALUES_LOAD_MODIFIER,           ///< Cache load modifier for reading vector values
    bool                            _DIRECT_LOAD_NONZEROS,                  ///< Whether to load nonzeros directly from global during sequential merging (vs. pre-staged through shared memory)
    BlockScanAlgorithm              _SCAN_ALGORITHM>                        ///< The BlockScan algorithm to use
struct AgentEasierPolicy
{
    enum
    {
        BLOCK_THREADS                                                   = _BLOCK_THREADS,                       ///< Threads per thread block
        ITEMS_PER_THREAD                                                = _ITEMS_PER_THREAD,                    ///< Items per thread (per tile of input)
        DIRECT_LOAD_NONZEROS                                            = _DIRECT_LOAD_NONZEROS,                ///< Whether to load nonzeros directly from global during sequential merging (pre-staged through shared memory)
        BATCH_SIZE                                                      = _BATCH_SIZE
    };

    static const CacheLoadModifier  ROW_OFFSETS_SEARCH_LOAD_MODIFIER    = _ROW_OFFSETS_SEARCH_LOAD_MODIFIER;    ///< Cache load modifier for reading CSR row-offsets
    static const CacheLoadModifier  ROW_OFFSETS_LOAD_MODIFIER           = _ROW_OFFSETS_LOAD_MODIFIER;           ///< Cache load modifier for reading CSR row-offsets
    static const CacheLoadModifier  COLUMN_INDICES_LOAD_MODIFIER        = _COLUMN_INDICES_LOAD_MODIFIER;        ///< Cache load modifier for reading CSR column-indices
    static const CacheLoadModifier  VALUES_LOAD_MODIFIER                = _VALUES_LOAD_MODIFIER;                ///< Cache load modifier for reading CSR values
    static const CacheLoadModifier  VECTOR_VALUES_LOAD_MODIFIER         = _VECTOR_VALUES_LOAD_MODIFIER;         ///< Cache load modifier for reading vector values
    static const BlockScanAlgorithm SCAN_ALGORITHM                      = _SCAN_ALGORITHM;                      ///< The BlockScan algorithm to use

};


/******************************************************************************
 * Thread block abstractions
 ******************************************************************************/

template <
    typename        ValueT,              ///< Matrix and vector value type
    typename        OffsetT>             ///< Signed integer type for sequence offsets
struct EasierParams
{
    #ifdef USE_LIST
    const ValueT**   d_values;            ///< Pointer to the array of \p num_nonzeros values of the corresponding nonzero elements of matrix <b>A</b>.
    const OffsetT*  d_row_end_offsets;   ///< Pointer to the array of \p m offsets demarcating the end of every row in \p d_column_indices and \p d_values
    const OffsetT**  d_column_indices;    ///< Pointer to the array of \p num_nonzeros column-indices of the corresponding nonzero elements of matrix <b>A</b>.  (Indices are zero-valued.)
    const ValueT**   d_vector_x;          ///< Pointer to the array of \p num_cols values corresponding to the dense input vector <em>x</em>
    #else
    const ValueT*   d_values;            ///< Pointer to the array of \p num_nonzeros values of the corresponding nonzero elements of matrix <b>A</b>.
    const OffsetT*  d_row_end_offsets;   ///< Pointer to the array of \p m offsets demarcating the end of every row in \p d_column_indices and \p d_values
    const OffsetT*  d_column_indices;    ///< Pointer to the array of \p num_nonzeros column-indices of the corresponding nonzero elements of matrix <b>A</b>.  (Indices are zero-valued.)
    const ValueT*   d_vector_x;          ///< Pointer to the array of \p num_cols values corresponding to the dense input vector <em>x</em>
    #endif
    ValueT*         d_vector_y;          ///< Pointer to the array of \p num_rows values corresponding to the dense output vector <em>y</em>
    int             num_rows;            ///< Number of rows of matrix <b>A</b>.
    int             num_cols;            ///< Number of columns of matrix <b>A</b>.
    int             num_nonzeros;        ///< Number of nonzero elements of matrix <b>A</b>.
    ValueT          alpha;               ///< Alpha multiplicand
    ValueT          beta;                ///< Beta addend-multiplicand
    int             batch_size;
};


/**
 * \brief AgentSpmv implements a stateful abstraction of CUDA thread blocks for participating in device-wide SpMV.
 */
template <
    typename    AgentEasierPolicyT,           ///< Parameterized AgentSpmvPolicy tuning policy type
    typename    ValueT,                     ///< Matrix and vector value type
    typename    OffsetT,                    ///< Signed integer type for sequence offsets
    bool        HAS_ALPHA,                  ///< Whether the input parameter \p alpha is 1
    bool        HAS_BETA,                   ///< Whether the input parameter \p beta is 0
    int         LEGACY_PTX_ARCH = 0>        ///< PTX compute capability (unused)
struct AgentEasier
{
    //---------------------------------------------------------------------
    // Types and constants
    //---------------------------------------------------------------------

    /// Constants
    enum
    {
        BLOCK_THREADS           = AgentEasierPolicyT::BLOCK_THREADS,
        ITEMS_PER_THREAD        = AgentEasierPolicyT::ITEMS_PER_THREAD,
        TILE_ITEMS              = BLOCK_THREADS * ITEMS_PER_THREAD,
        BATCH_SIZE              = AgentEasierPolicyT::BATCH_SIZE,
    };

    /// 2D merge path coordinate type
    typedef typename CubVector<OffsetT, 2>::Type CoordinateT;

    /// Input iterator wrapper types (for applying cache modifiers)

    typedef CacheModifiedInputIterator<
            AgentEasierPolicyT::ROW_OFFSETS_SEARCH_LOAD_MODIFIER,
            OffsetT,
            OffsetT>
        RowOffsetsSearchIteratorT;

    typedef CacheModifiedInputIterator<
            AgentEasierPolicyT::ROW_OFFSETS_LOAD_MODIFIER,
            OffsetT,
            OffsetT>
        RowOffsetsIteratorT;

    #ifdef USE_LIST
    typedef CacheModifiedInputIterator<
        AgentEasierPolicyT::COLUMN_INDICES_LOAD_MODIFIER,
        const OffsetT*,
        OffsetT>
    ColumnIndicesIteratorT;

    typedef CacheModifiedInputIterator<
            AgentEasierPolicyT::VALUES_LOAD_MODIFIER,
            const ValueT*,
            OffsetT>
        ValueIteratorT;

    typedef CacheModifiedInputIterator<
            AgentEasierPolicyT::VECTOR_VALUES_LOAD_MODIFIER,
            const ValueT*,
            OffsetT>
        VectorValueIteratorTx;
    #else
    typedef CacheModifiedInputIterator<
            AgentEasierPolicyT::COLUMN_INDICES_LOAD_MODIFIER,
            OffsetT,
            OffsetT>
        ColumnIndicesIteratorT;

    typedef CacheModifiedInputIterator<
            AgentEasierPolicyT::VALUES_LOAD_MODIFIER,
            ValueT,
            OffsetT>
        ValueIteratorT;

    typedef CacheModifiedInputIterator<
            AgentEasierPolicyT::VECTOR_VALUES_LOAD_MODIFIER,
            ValueT,
            OffsetT>
        VectorValueIteratorTx;
    #endif

    typedef CacheModifiedInputIterator<
            AgentEasierPolicyT::VECTOR_VALUES_LOAD_MODIFIER,
            ValueT,
            OffsetT>
        VectorValueIteratorTy;

    // Tuple type for scanning (pairs accumulated segment-value with segment-index)
    typedef KeyValuePair<OffsetT, ValueT> KeyValuePairT;

    // Reduce-value-by-segment scan operator
    typedef ReduceByKeyOp<cub::Sum> ReduceBySegmentOpT;

    // BlockReduce specialization
    typedef BlockReduce<
            ValueT,
            BLOCK_THREADS,
            BLOCK_REDUCE_WARP_REDUCTIONS>
        BlockReduceT;

    // BlockScan specialization
    typedef BlockScan<
            KeyValuePairT,
            BLOCK_THREADS,
            AgentEasierPolicyT::SCAN_ALGORITHM>
        BlockScanT;

    // BlockScan specialization
    typedef BlockScan<
            ValueT,
            BLOCK_THREADS,
            AgentEasierPolicyT::SCAN_ALGORITHM>
        BlockPrefixSumT;

    // BlockExchange specialization
    typedef BlockExchange<
            ValueT,
            BLOCK_THREADS,
            ITEMS_PER_THREAD>
        BlockExchangeT;

    /// Merge item type (either a non-zero value or a row-end offset)
    union MergeItem
    {
      // Value type to pair with index type OffsetT
      // (NullType if loading values directly during merge)
      using MergeValueT =
        cub::detail::conditional_t<
          AgentEasierPolicyT::DIRECT_LOAD_NONZEROS, NullType, ValueT>;

      OffsetT row_end_offset;
      MergeValueT nonzero;
    };

    /// Shared memory type required by this thread block
    #include "my_shared_memory.cuh"

    /// Temporary storage type (unionable)
    struct TempStorage : Uninitialized<_TempStorage> {};


    //---------------------------------------------------------------------
    // Per-thread fields
    //---------------------------------------------------------------------


    _TempStorage&                   temp_storage;         /// Reference to temp_storage

    EasierParams<ValueT, OffsetT>&    easier_params;

    ValueIteratorT                  wd_values;            ///< Wrapped pointer to the array of \p num_nonzeros values of the corresponding nonzero elements of matrix <b>A</b>.
    RowOffsetsIteratorT             wd_row_end_offsets;   ///< Wrapped Pointer to the array of \p m offsets demarcating the end of every row in \p d_column_indices and \p d_values
    ColumnIndicesIteratorT          wd_column_indices;    ///< Wrapped Pointer to the array of \p num_nonzeros column-indices of the corresponding nonzero elements of matrix <b>A</b>.  (Indices are zero-valued.)
    VectorValueIteratorTx            wd_vector_x;          ///< Wrapped Pointer to the array of \p num_cols values corresponding to the dense input vector <em>x</em>
    VectorValueIteratorTy            wd_vector_y;          ///< Wrapped Pointer to the array of \p num_cols values corresponding to the dense input vector <em>x</em>


    //---------------------------------------------------------------------
    // Interface
    //---------------------------------------------------------------------

    /**
     * Constructor
     */
    __device__ __forceinline__ AgentEasier(
        TempStorage&                    temp_storage,           ///< Reference to temp_storage
        EasierParams<ValueT, OffsetT>&    easier_params)            ///< SpMV input parameter bundle
    :
        temp_storage(temp_storage.Alias()),
        easier_params(easier_params),
        wd_values(easier_params.d_values),
        wd_row_end_offsets(easier_params.d_row_end_offsets),
        wd_column_indices(easier_params.d_column_indices),
        wd_vector_x(easier_params.d_vector_x),
        wd_vector_y(easier_params.d_vector_y)
    {}


    __device__ __forceinline__ void ConsumeTile_b(
        int             tile_idx,
        CoordinateT     tile_start_coord,
        CoordinateT     tile_end_coord,
        KeyValuePairT*  d_tile_carry_pairs,
        Int2Type<false> is_direct_load)     ///< Marker type indicating whether to load nonzeros directly during path-discovery or beforehand in batch
    {
        int         tile_num_rows           = tile_end_coord.x - tile_start_coord.x;
        int         tile_num_nonzeros       = tile_end_coord.y - tile_start_coord.y;

        //temp_storage.aliasable.merge_items分成了两个部分一部分存row_end_offset(前半部分)，一部分存tile_nonzeros(后半部分)
        OffsetT*    s_tile_row_end_offsets  = &temp_storage.aliasable.batch_op.merge_items[0].row_end_offset;//前面的tile_num_rows + ITEMS_PER_THREAD项的对应的空间用于保存reduction?
        ValueT*     s_tile_nonzeros         = &temp_storage.aliasable.batch_op.merge_items[(tile_num_rows + ITEMS_PER_THREAD)*BATCH_SIZE].nonzero;//先计算乘法并保存到shared memory当中
        ValueT*     s_input_buffer          = &temp_storage.input_buffer_0[0];
        // Gather the nonzeros for the merge tile into shared memory


        //version 1:use param to give batch info
        // int BATCH_LIST[2] = { 3, BATCH_SIZE};
        int BATCH_LIST[1] = {BATCH_SIZE};
        MyStruct::compute_before_scatter(ITEMS_PER_THREAD,
                                         BLOCK_THREADS,
                                         tile_num_nonzeros,
                                         easier_params.num_nonzeros,
                                         easier_params.num_rows,
                                         tile_start_coord,
                                         BATCH_LIST,
                                        //  2,
                                         1,
                                         wd_values,
                                         wd_column_indices,
                                         wd_vector_x,
                                         s_input_buffer,
                                         s_tile_nonzeros);

        // //version 2:only handle batch info when auto code generation
        // MyStruct::compute_before_scatter_simple(ITEMS_PER_THREAD,
        //                                         BLOCK_THREADS,
        //                                         tile_num_nonzeros,
        //                                         easier_params.num_nonzeros,
        //                                         easier_params.num_rows,
        //                                         BATCH_LIST,
        //                                         2,
        //                                         wd_values,
        //                                         wd_column_indices,
        //                                         wd_vector_x,
        //                                         tile_start_coord,
        //                                         s_tile_nonzeros);


        // #pragma unroll
        // for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        // {
        //     int nonzero_idx = threadIdx.x + (ITEM * BLOCK_THREADS);
        //     for (int batch_idx=0; batch_idx < BATCH_SIZE; ++batch_idx) {
        //         ValueIteratorT a                = wd_values + tile_start_coord.y + nonzero_idx + batch_idx * easier_params.num_nonzeros ;
        //         ColumnIndicesIteratorT ci       = wd_column_indices + tile_start_coord.y + nonzero_idx;
        //         ValueT* s                       = s_tile_nonzeros + nonzero_idx * BATCH_SIZE + batch_idx;

        //         if (nonzero_idx < tile_num_nonzeros)
        //         {

        //             OffsetT column_idx              = *ci;
        //             ValueT  value                   = *a;

        //             ValueT  vector_value            = wd_vector_x[column_idx * BATCH_SIZE + batch_idx];

        //             ValueT  nonzero                 = value * vector_value;

        //             *s    = nonzero;
        //         }
        //     }
        // }


        // Gather the row end-offsets for the merge tile into shared memory
        #pragma unroll 1
        for (int item = threadIdx.x; item < tile_num_rows + ITEMS_PER_THREAD; item += BLOCK_THREADS)
        {
            const OffsetT offset =
              (cub::min)(static_cast<OffsetT>(tile_start_coord.x + item),
                         static_cast<OffsetT>(easier_params.num_rows - 1));
            s_tile_row_end_offsets[item] = wd_row_end_offsets[offset];//设置s_tile_row_end_offsets
        }

        CTA_SYNC();

        // Search for the thread's starting coordinate within the merge tile
        CountingInputIterator<OffsetT>  tile_nonzero_indices(tile_start_coord.y);
        CoordinateT                     thread_start_coord;

        MergePathSearch(
            OffsetT(threadIdx.x * ITEMS_PER_THREAD),    // Diagonal
            s_tile_row_end_offsets,                     // List A row_end_offsets 的 substring
            tile_nonzero_indices,                       // List B nnz 的 substring
            tile_num_rows,
            tile_num_nonzeros,
            thread_start_coord);

        CTA_SYNC();            // Perf-sync

        // Compute the thread's merge path segment
        CoordinateT     thread_current_coord = thread_start_coord;
        KeyValuePairT   scan_segment[BATCH_SIZE][ITEMS_PER_THREAD];  //记录每一个item的multiplication的部分累加结果以及对应的row的index值

        ValueT          running_total[BATCH_SIZE];
        #pragma unroll
        for (int batch_idx=0; batch_idx < BATCH_SIZE; ++batch_idx) {
            running_total[batch_idx]    = 0.0;
        }

        OffsetT row_end_offset  = s_tile_row_end_offsets[thread_current_coord.x];
        ValueT  nonzero         = s_tile_nonzeros[thread_current_coord.y * BATCH_SIZE];  //FIXME:may cause bank conflict

        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
        {
            if (tile_nonzero_indices[thread_current_coord.y] < row_end_offset)
            {
                #pragma unroll
                for (int batch_idx=0; batch_idx < BATCH_SIZE; ++batch_idx) {
                    // Move down (accumulate)
                    scan_segment[batch_idx][ITEM].value    = nonzero;
                    running_total[batch_idx]               += nonzero;
                    nonzero                     = s_tile_nonzeros[thread_current_coord.y * BATCH_SIZE + batch_idx + 1];
                }
                ++thread_current_coord.y;
                nonzero                     = s_tile_nonzeros[thread_current_coord.y * BATCH_SIZE];
            }
            else
            {
                // Move right (reset)
                #pragma unroll
                for (int batch_idx=0; batch_idx < BATCH_SIZE; ++batch_idx) {
                    scan_segment[batch_idx][ITEM].value    = 0.0;
                    running_total[batch_idx]               = 0.0;
                }
                ++thread_current_coord.x;
                row_end_offset              = s_tile_row_end_offsets[thread_current_coord.x];
            }
            #pragma unroll
            for (int batch_idx=0; batch_idx < BATCH_SIZE; ++batch_idx) {
                scan_segment[batch_idx][ITEM].key = thread_current_coord.x;
            }
        }

        CTA_SYNC();

        // Block-wide reduce-value-by-segment
        
        ReduceBySegmentOpT  scan_op;
        KeyValuePairT       scan_item[BATCH_SIZE];
        KeyValuePairT       tile_carry[BATCH_SIZE];

        #pragma unroll
        for (int batch_idx=0; batch_idx < BATCH_SIZE; ++batch_idx) {
            scan_item[batch_idx].value = running_total[batch_idx];//每一个thread负责的最后一行（不完整）的累加值
            scan_item[batch_idx].key   = thread_current_coord.x;//每一个thread负责的最后一行（不完整）在s_tile_row_end_offsets中的index
            BlockScanT(temp_storage.aliasable.scan).ExclusiveScan(scan_item[batch_idx], scan_item[batch_idx], scan_op, tile_carry[batch_idx]);//reduce by key，拿到了这个block负责的最后一行的partial sum以及index
            CTA_SYNC();
        }

        if (threadIdx.x == 0)
        {
            #pragma unroll
            for (int batch_idx=0; batch_idx < BATCH_SIZE; ++batch_idx) {
                scan_item[batch_idx].key = thread_start_coord.x;
                scan_item[batch_idx].value = 0.0;
            }
        }

        if (tile_num_rows > 0)
        {

            CTA_SYNC();

            // Scan downsweep and scatter
            ValueT* s_partials = &temp_storage.aliasable.batch_op.merge_items[0].nonzero;

            if (scan_item[0].key != scan_segment[0][0].key)
            {
                #pragma unroll
                for (int batch_idx=0; batch_idx < BATCH_SIZE; ++batch_idx) {
                    s_partials[scan_item[0].key * BATCH_SIZE + batch_idx] = scan_item[batch_idx].value;
                }
            }
            else
            {
                #pragma unroll
                for (int batch_idx=0; batch_idx < BATCH_SIZE; ++batch_idx) {
                    scan_segment[batch_idx][0].value += scan_item[batch_idx].value;
                }
            }

            #pragma unroll
            for (int ITEM = 1; ITEM < ITEMS_PER_THREAD; ++ITEM)
            {
                if (scan_segment[0][ITEM - 1].key != scan_segment[0][ITEM].key)
                {
                    #pragma unroll
                    for (int batch_idx=0; batch_idx < BATCH_SIZE; ++batch_idx) {
                        s_partials[scan_segment[batch_idx][ITEM - 1].key * BATCH_SIZE + batch_idx] = scan_segment[batch_idx][ITEM - 1].value;
                    }
                }
                else
                {
                    #pragma unroll
                    for (int batch_idx=0; batch_idx < BATCH_SIZE; ++batch_idx) {
                        scan_segment[batch_idx][ITEM].value += scan_segment[batch_idx][ITEM - 1].value;
                    }
                }
            }

            CTA_SYNC();

            #pragma unroll 1
            for (int item = threadIdx.x; item < tile_num_rows; item += BLOCK_THREADS)
            {
                #pragma unroll
                for (int batch_idx=0; batch_idx < BATCH_SIZE; ++batch_idx) {
                    easier_params.d_vector_y[(tile_start_coord.x + item) + batch_idx * easier_params.num_rows] = s_partials[item * BATCH_SIZE + batch_idx];//直接store到global memory
                }
            }
        }

        // Output the tile's carry-out
        if (threadIdx.x == 0)//只有threadIdx.x == 0拿到的是这个block最后一行的最前面一部分的partial sum
        {
            #pragma unroll
            for (int batch_idx=0; batch_idx < BATCH_SIZE; ++batch_idx) {
                {
                if (HAS_ALPHA)
                    tile_carry[batch_idx].value *= easier_params.alpha;
                }

                tile_carry[batch_idx].key += tile_start_coord.x;//得到实际的row index
                if (tile_carry[batch_idx].key >= easier_params.num_rows)
                {
                    // FIXME: This works around an invalid memory access in the
                    // fixup kernel. The underlying issue needs to be debugged and
                    // properly fixed, but this hack prevents writes to
                    // out-of-bounds addresses. It doesn't appear to have an effect
                    // on the validity of the results, since this only affects the
                    // carry-over from last tile in the input.
                    tile_carry[batch_idx].key = easier_params.num_rows - 1;
                    tile_carry[batch_idx].value = ValueT{};
                };
                if (tile_carry[batch_idx].key >= easier_params.num_rows) {
                    printf("find over easier_params.num_rows for batch_idx:%d\n",batch_idx);

                }

                // if(batch_idx!=0)
                d_tile_carry_pairs[tile_idx + batch_idx * gridDim.x]    = tile_carry[batch_idx];//写回到记录部分和的global memory当中
            }

        }

        // Return the tile's running carry-out
        return;
    }




    /**
     * Consume input tile
     */
    __device__ __forceinline__ void ConsumeTile(
        CoordinateT*    d_tile_coordinates,     ///< [in] Pointer to the temporary array of tile starting coordinates
        KeyValuePairT*  d_tile_carry_pairs,     ///< [out] Pointer to the temporary array carry-out dot product row-ids, one per block
        int             num_merge_tiles)        ///< [in] Number of merge tiles
    {
        int tile_idx = (blockIdx.x * gridDim.y) + blockIdx.y;    // Current tile index

        if (tile_idx >= num_merge_tiles)
            return;

        // Read our starting coordinates
        if (threadIdx.x < 2)
        {
            if (d_tile_coordinates == NULL)
            {
                // Search our starting coordinates
                OffsetT                         diagonal = (tile_idx + threadIdx.x) * TILE_ITEMS;
                CoordinateT                     tile_coord;
                CountingInputIterator<OffsetT>  nonzero_indices(0);

                // Search the merge path
                MergePathSearch(
                    diagonal,
                    RowOffsetsSearchIteratorT(easier_params.d_row_end_offsets),
                    nonzero_indices,
                    easier_params.num_rows,
                    easier_params.num_nonzeros,
                    tile_coord);

                temp_storage.tile_coords[threadIdx.x] = tile_coord;
            }
            else
            {
                temp_storage.tile_coords[threadIdx.x] = d_tile_coordinates[tile_idx + threadIdx.x];//拿到当前tile block的起始position以及结束的position
            }
        }

        CTA_SYNC();

        CoordinateT tile_start_coord     = temp_storage.tile_coords[0];//每一个thread拿到起始的position
        CoordinateT tile_end_coord       = temp_storage.tile_coords[1];
        // Consume multi-segment tile
        ConsumeTile_b(
            tile_idx,
            tile_start_coord,
            tile_end_coord,
            d_tile_carry_pairs,
            Int2Type<AgentEasierPolicyT::DIRECT_LOAD_NONZEROS>());//false here
    }


};




CUB_NAMESPACE_END

