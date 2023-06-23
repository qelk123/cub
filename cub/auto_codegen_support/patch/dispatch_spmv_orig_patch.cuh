
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
 * cub::DeviceSpmv provides device-wide parallel operations for performing sparse-matrix * vector multiplication (SpMV).
 */

#pragma once

#include <cub/agent/agent_segment_fixup.cuh>
#include "./agent_spmv_orig_patch.cuh"
#include <cub/agent/single_pass_scan_operators.cuh>
#include <cub/config.cuh>
#include <cub/detail/cpp_compatibility.cuh>
#include <cub/grid/grid_queue.cuh>
#include <cub/thread/thread_search.cuh>
#include <cub/util_debug.cuh>
#include <cub/util_deprecated.cuh>
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>
#include <cub/util_type.cuh>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

#include <cstdio>
#include <iterator>

#include <nv/target>

CUB_NAMESPACE_BEGIN


/******************************************************************************
 * SpMV kernel entry points
 *****************************************************************************/


/**
 * Spmv search kernel. Identifies merge path starting coordinates for each tile.
 */
template <
    typename    EasierPolicyT,                    ///< Parameterized SpmvPolicy tuning policy type
    typename    OffsetT,                        ///< Signed integer type for sequence offsets
    typename    CoordinateT,                    ///< Merge path coordinate type
    typename    EasierParamsT>                    ///< EasierParams type
__global__ void MyDeviceSpmvSearchKernel(
    int             num_merge_tiles,            ///< [in] Number of SpMV merge tiles (spmv grid size)
    CoordinateT*    d_tile_coordinates,         ///< [out] Pointer to the temporary array of tile starting coordinates
    EasierParamsT     easier_params)                ///< [in] SpMV input parameter bundle
{
    /// Constants
    enum
    {
        BLOCK_THREADS           = EasierPolicyT::BLOCK_THREADS,
        ITEMS_PER_THREAD        = EasierPolicyT::ITEMS_PER_THREAD,
        TILE_ITEMS              = BLOCK_THREADS * ITEMS_PER_THREAD,
    };

    typedef CacheModifiedInputIterator<
            EasierPolicyT::ROW_OFFSETS_SEARCH_LOAD_MODIFIER,
            OffsetT,
            OffsetT>
        RowOffsetsSearchIteratorT;

    // Find the starting coordinate for all tiles (plus the end coordinate of the last one)
    int tile_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tile_idx < num_merge_tiles + 1)
    {
        OffsetT                         diagonal = (tile_idx * TILE_ITEMS);//diagonal通过均分得到
        CoordinateT                     tile_coordinate;
        CountingInputIterator<OffsetT>  nonzero_indices(0);

        // Search the merge path
        MergePathSearch(//根据diagonal进行二分搜索
            diagonal,
            RowOffsetsSearchIteratorT(easier_params.d_row_end_offsets),
            nonzero_indices,
            easier_params.num_rows,
            easier_params.num_nonzeros,
            tile_coordinate);

        // Output starting offset
        d_tile_coordinates[tile_idx] = tile_coordinate;//拿到每一个block对应的coordinate
    }
}


/**
 * Spmv agent entry point
 */
template <
    typename        EasierPolicyT,                ///< Parameterized SpmvPolicy tuning policy type
    typename        ScanTileStateT,             ///< Tile status interface type
    typename        ValueT,                     ///< Matrix and vector value type
    typename        OffsetT,                    ///< Signed integer type for sequence offsets
    typename        CoordinateT,                ///< Merge path coordinate type
    bool            HAS_ALPHA,                  ///< Whether the input parameter Alpha is 1
    bool            HAS_BETA>                   ///< Whether the input parameter Beta is 0
__launch_bounds__ (int(EasierPolicyT::BLOCK_THREADS))
__global__ void MyDeviceSpmvKernel(
    EasierParams<ValueT, OffsetT>   easier_params,                ///< [in] SpMV input parameter bundle
    CoordinateT*                    d_tile_coordinates,         ///< [in] Pointer to the temporary array of tile starting coordinates
    // void*   d_tile_carry_pairs,         ///< [out] Pointer to the temporary array carry-out dot product row-ids, one per block
    KeyValuePair<OffsetT,ValueT>**   d_tile_carry_pairs,         ///< [out] Pointer to the temporary array carry-out dot product row-ids, one per block
    // KeyValuePair<OffsetT,ValueT>*   d_tile_carry_pairs,         ///< [out] Pointer to the temporary array carry-out dot product row-ids, one per block
    int                             num_tiles,                  ///< [in] Number of merge tiles
    #ifdef USE_MULTI_STREAM_PPOST
    void*                           tile_state,                 ///< [in] Tile status interface for fixup reduce-by-key kernel
    #else
    ScanTileStateT                  tile_state,
    #endif
    int                             batch_size,                 ///< [in] Tile status interface for fixup reduce-by-key kernel
    int                             num_segment_fixup_tiles)    ///< [in] Number of reduce-by-key tiles (fixup grid size)
{
    // Spmv agent type specialization
    typedef AgentEasier<
            EasierPolicyT,
            ValueT,
            OffsetT,
            HAS_ALPHA,
            HAS_BETA>
        AgentEasierT;

    // Shared memory for AgentSpmv
    __shared__ typename AgentEasierT::TempStorage temp_storage;

    AgentEasierT(temp_storage, easier_params).ConsumeTile(
        d_tile_coordinates,
        d_tile_carry_pairs,
        num_tiles);

    // Initialize fixup tile status
    #ifdef USE_MULTI_STREAM_PPOST
    for (int i = 0; i < batch_size; i++)
        ((ScanTileStateT *)tile_state)[i].InitializeStatus(num_segment_fixup_tiles);//only initialize
    #else
        tile_state.InitializeStatus(num_segment_fixup_tiles);//only initialize
    #endif
}

/**
 * Multi-block reduce-by-key sweep kernel entry point
 */
template <
    typename    AgentSegmentFixupPolicyT,       ///< Parameterized AgentSegmentFixupPolicy tuning policy type
    typename    PairsInputIteratorT,            ///< Random-access input iterator type for keys
    typename    AggregatesOutputIteratorT,      ///< Random-access output iterator type for values
    typename    OffsetT,                        ///< Signed integer type for global offsets
    typename    ScanTileStateT>                 ///< Tile status interface type
__launch_bounds__ (int(AgentSegmentFixupPolicyT::BLOCK_THREADS))
__global__ void MyDeviceSegmentFixupKernel(
    PairsInputIteratorT         d_pairs_in,         ///< [in] Pointer to the array carry-out dot product row-ids, one per spmv block
    AggregatesOutputIteratorT   d_aggregates_out,   ///< [in,out] Output value aggregates
    OffsetT                     num_items,          ///< [in] Total number of items to select from
    int                         num_tiles,          ///< [in] Total number of tiles for the entire problem
    ScanTileStateT              tile_state)         ///< [in] Tile status interface         
{
    // Thread block type for reducing tiles of value segments
    typedef AgentSegmentFixup<
            AgentSegmentFixupPolicyT,
            PairsInputIteratorT,
            AggregatesOutputIteratorT,
            cub::Equality,
            cub::Sum,
            OffsetT>
        AgentSegmentFixupT;

    // Shared memory for AgentSegmentFixup
    __shared__ typename AgentSegmentFixupT::TempStorage temp_storage;

    // Process tiles
    AgentSegmentFixupT(temp_storage, d_pairs_in, d_aggregates_out, cub::Equality(), cub::Sum()).ConsumeRange(
        num_items,
        num_tiles,
        tile_state);
    
}

/**
 * Multi-block reduce-by-key sweep kernel entry point
 */
template <
    typename    AgentTileStateInitPolicyT,       ///< Parameterized AgentSegmentFixupPolicy tuning policy type
    typename    ScanTileStateT>                 ///< Tile status interface type
__launch_bounds__ (int(AgentTileStateInitPolicyT::BLOCK_THREADS))
__global__ void MyDeviceTileStateInitKernel(
    int                         num_segment_fixup_tiles,          ///< [in] Total number of tiles for the entire problem
    ScanTileStateT              tile_state         ///< [in] Tile status interface
    )         
{

        tile_state.InitializeStatus(num_segment_fixup_tiles);//only initialize
    
}


/******************************************************************************
 * Dispatch
 ******************************************************************************/

/**
 * Utility class for dispatching the appropriately-tuned kernels for DeviceSpmv
 */
template <
    typename    ValueT,                     ///< Matrix and vector value type
    typename    OffsetT,
    int         BatchSize>                    ///< Signed integer type for global offsets
struct DispatchEasier
{
    //---------------------------------------------------------------------
    // Constants and Types
    //---------------------------------------------------------------------

    enum
    {
        INIT_KERNEL_THREADS = 128,
        EMPTY_MATRIX_KERNEL_THREADS = 128
    };

    // EasierParams bundle type
    typedef EasierParams<ValueT, OffsetT> EasierParamsT;

    // 2D merge path coordinate type
    typedef typename CubVector<OffsetT, 2>::Type CoordinateT;

    // Tile status descriptor interface type
    typedef ReduceByKeyScanTileState<ValueT, OffsetT> ScanTileStateT;

    // Tuple type for scanning (pairs accumulated segment-value with segment-index)
    typedef KeyValuePair<OffsetT, ValueT> KeyValuePairT;


    //---------------------------------------------------------------------
    // Tuning policies
    //---------------------------------------------------------------------


    /// SM60
    struct Policy600
    {
        typedef AgentEasierPolicy<
                BLOCK_SIZE,
                ITEM_PER_THREAD,
                LOAD_DEFAULT,
                LOAD_DEFAULT,
                LOAD_DEFAULT,
                LOAD_DEFAULT,
                LOAD_DEFAULT,
                false,
                BLOCK_SCAN_WARP_SCANS>
            EasierPolicyT;


        typedef AgentSegmentFixupPolicy<
                128,
                3,
                BLOCK_LOAD_DIRECT,
                LOAD_LDG,
                BLOCK_SCAN_WARP_SCANS>
            SegmentFixupPolicyT;


        struct ScanTileStatePolicyT
        {
            enum
            {
                BLOCK_THREADS           = 128,               ///< Threads per thread block
                ITEMS_PER_THREAD        = 1,            ///< Items per thread (per tile of input)
            };
        };
    };



    //---------------------------------------------------------------------
    // Tuning policies of current PTX compiler pass
    //---------------------------------------------------------------------

    typedef Policy600 PtxPolicy;


    // "Opaque" policies (whose parameterizations aren't reflected in the type signature)
    struct PtxSpmvPolicyT : PtxPolicy::EasierPolicyT {};
    struct PtxSegmentFixupPolicy : PtxPolicy::SegmentFixupPolicyT {};
    struct PtxTileStateInitPolicy : PtxPolicy::ScanTileStatePolicyT {};


    //---------------------------------------------------------------------
    // Utilities
    //---------------------------------------------------------------------

    /**
     * Initialize kernel dispatch configurations with the policies corresponding to the PTX assembly we will use
     */
    template <typename KernelConfig>
    CUB_RUNTIME_FUNCTION __forceinline__
    static void InitConfigs(
        int             ptx_version,
        KernelConfig    &spmv_config,
        KernelConfig    &segment_fixup_config,
        KernelConfig    &tile_state_init_config)
    {
      NV_IF_TARGET(
        NV_IS_DEVICE,
        ( // We're on the device, so initialize the kernel dispatch
          // configurations with the current PTX policy
          spmv_config.template Init<PtxSpmvPolicyT>();
          segment_fixup_config.template Init<PtxSegmentFixupPolicy>();
          tile_state_init_config.template Init<PtxTileStateInitPolicy>();),
        (
            spmv_config.template Init<typename Policy600::EasierPolicyT>();

            segment_fixup_config.template Init<typename Policy600::SegmentFixupPolicyT>();

            tile_state_init_config.template Init<typename Policy600::ScanTileStatePolicyT>();
        ));
    }


    /**
     * Kernel kernel dispatch configuration.
     */
    struct KernelConfig
    {
        int block_threads;
        int items_per_thread;
        int tile_items;

        template <typename PolicyT>
        CUB_RUNTIME_FUNCTION __forceinline__
        void Init()
        {
            block_threads       = PolicyT::BLOCK_THREADS;
            items_per_thread    = PolicyT::ITEMS_PER_THREAD;
            tile_items          = block_threads * items_per_thread;
        }
    };


    //---------------------------------------------------------------------
    // Dispatch entrypoints
    //---------------------------------------------------------------------

    /**
     * Internal dispatch routine for computing a device-wide reduction using the
     * specified kernel functions.
     *
     * If the input is larger than a single tile, this method uses two-passes of
     * kernel invocations.
     */
    template <
        // typename                Spmv1ColKernelT,                    ///< Function type of cub::MyDeviceSpmv1ColKernel
        typename                SpmvSearchKernelT,                  ///< Function type of cub::AgentSpmvSearchKernel
        typename                SpmvKernelT,                        ///< Function type of cub::AgentSpmvKernel
        typename                SegmentFixupKernelT,                ///< Function type of cub::DeviceSegmentFixupKernelT
        // typename                SpmvEmptyMatrixKernelT,             ///< Function type of cub::MyDeviceSpmvEmptyMatrixKernel
        typename                TileStateInitKernelT>
    CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t Dispatch(
        void*                   d_temp_storage,                     ///< [in] Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t&                 temp_storage_bytes,                 ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        EasierParamsT&          easier_params,                        ///< SpMV input parameter bundle
        cudaStream_t            stream,                             ///< [in] CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        SpmvSearchKernelT       spmv_search_kernel,                 ///< [in] Kernel function pointer to parameterization of AgentSpmvSearchKernel
        SpmvKernelT             spmv_kernel,                        ///< [in] Kernel function pointer to parameterization of AgentSpmvKernel
        SegmentFixupKernelT     segment_fixup_kernel,               ///< [in] Kernel function pointer to parameterization of cub::MyDeviceSegmentFixupKernel
        TileStateInitKernelT    tile_state_init_kernel,
        KernelConfig            spmv_config,                        ///< [in] Dispatch parameters that match the policy that \p spmv_kernel was compiled for
        KernelConfig            segment_fixup_config,               ///< [in] Dispatch parameters that match the policy that \p segment_fixup_kernel was compiled for
        KernelConfig            tile_state_init_config)               
    {
        cudaError error = cudaSuccess;
        do
        {
            if (easier_params.num_rows < 0 || easier_params.num_cols < 0)
            {
              return cudaErrorInvalidValue;
            }

            if (easier_params.num_rows == 0 || easier_params.num_cols == 0)
            { // Empty problem, no-op.
                if (d_temp_storage == NULL)
                {
                    temp_storage_bytes = 1;
                }

                break;
            }

            // Get device ordinal
            int device_ordinal;
            if (CubDebug(error = cudaGetDevice(&device_ordinal))) break;

            // Get SM count
            int sm_count;
            if (CubDebug(error = cudaDeviceGetAttribute (&sm_count, cudaDevAttrMultiProcessorCount, device_ordinal))) break;

            // Get max x-dimension of grid
            int max_dim_x;
            if (CubDebug(error = cudaDeviceGetAttribute(&max_dim_x, cudaDevAttrMaxGridDimX, device_ordinal))) break;

            // Total number of spmv work items
            int num_merge_items = easier_params.num_rows + easier_params.num_nonzeros;//用行数加上非零元数

            // Tile sizes of kernels
            int merge_tile_size              = spmv_config.block_threads * spmv_config.items_per_thread;//一个block handle的item
            int segment_fixup_tile_size     = segment_fixup_config.block_threads * segment_fixup_config.items_per_thread;
            int init_state_tile_size       = tile_state_init_config.block_threads * tile_state_init_config.items_per_thread;//一个block handle的item

            // Number of tiles for kernels
            int num_merge_tiles            = cub::DivideAndRoundUp(num_merge_items, merge_tile_size);//block的个数
            int num_segment_fixup_tiles    = cub::DivideAndRoundUp(num_merge_tiles, segment_fixup_tile_size);
            int num_state_init_tiles       = cub::DivideAndRoundUp(num_merge_tiles, init_state_tile_size);





            // Get SM occupancy for kernels
            int spmv_sm_occupancy;
            if (CubDebug(error = MaxSmOccupancy(
                spmv_sm_occupancy,
                spmv_kernel,
                spmv_config.block_threads))) break;

            int segment_fixup_sm_occupancy;
            if (CubDebug(error = MaxSmOccupancy(
                segment_fixup_sm_occupancy,
                segment_fixup_kernel,
                segment_fixup_config.block_threads))) break;

            int tile_state_init_sm_occupancy;
            if (CubDebug(error = MaxSmOccupancy(
                tile_state_init_sm_occupancy,
                tile_state_init_kernel,
                tile_state_init_config.block_threads))) break;

            // Get grid dimensions
            dim3 spmv_grid_size(
                CUB_MIN(num_merge_tiles, max_dim_x),
                cub::DivideAndRoundUp(num_merge_tiles, max_dim_x),
                1);

            dim3 segment_fixup_grid_size(
                CUB_MIN(num_segment_fixup_tiles, max_dim_x),
                cub::DivideAndRoundUp(num_segment_fixup_tiles, max_dim_x),
                1);

            dim3 tile_init_state_grid_size(
                CUB_MIN(num_state_init_tiles, max_dim_x),
                cub::DivideAndRoundUp(num_state_init_tiles, max_dim_x),
                1);

            // Get the temporary storage allocation requirements
            #ifdef USE_MULTI_STREAM_PPOST
                size_t allocation_sizes[BatchSize + 3];
                for (int i = 0; i < BatchSize ; i++) {
                    if (CubDebug(error = ScanTileStateT::AllocationSize(num_segment_fixup_tiles, allocation_sizes[i]))) break;    // bytes needed for reduce-by-key tile status descriptors
                }
                    // allocation_sizes[0] =  (num_tiles + TILE_STATUS_PADDING) * sizeof(TxnWord)
                allocation_sizes[BatchSize] = BatchSize * sizeof(ScanTileStateT *);       // bytes needed for block carry-out pairs
                
                allocation_sizes[BatchSize + 1] = num_merge_tiles * BatchSize * sizeof(KeyValuePairT);       // bytes needed for block carry-out pairs
                allocation_sizes[BatchSize + 2] = (num_merge_tiles + 1) * sizeof(CoordinateT);   // bytes needed for tile starting coordinates

                // Alias the temporary allocations from the single storage blob (or compute the necessary size of the blob)
                void* allocations[BatchSize + 3] = {};
                if (CubDebug(error = AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes))) break;
                //从[d_temp_storage,d_temp_storage+temp_storage_bytes)分配合计allocation_sizes的空间并返回对应到每一个global buffer的指针放进allocations中返回
                if (d_temp_storage == NULL)
                {
                    // Return if the caller is simply requesting the size of the storage allocation
                    break;
                }

                // Construct the tile status interface



                ScanTileStateT tile_state[BatchSize];
                for (int i = 0; i < BatchSize ; i++) {
                    if (CubDebug(error = tile_state[i].Init(num_segment_fixup_tiles, allocations[i], allocation_sizes[i]))) break;
                }

                CubDebugExit(cudaMemcpy((void *)allocations[BatchSize], (const void *)tile_state, BatchSize * sizeof(ScanTileStateT *), cudaMemcpyHostToDevice));


                // Alias the other allocations
                ScanTileStateT* tile_state_list         = (ScanTileStateT*) allocations[BatchSize];  // Agent carry-out pairs
                KeyValuePairT*  d_tile_carry_pairs      = (KeyValuePairT*) allocations[BatchSize + 1];  // Agent carry-out pairs
                CoordinateT*    d_tile_coordinates      = (CoordinateT*) allocations[BatchSize + 2];    // Agent starting coordinates
            #else
                // #include "dispatch_helper_patch.cuh"
                #include <dispatch_helper.cuh>
                
                size_t allocation_sizes[scatter_op_num + 3];
                for (int i = 0; i < scatter_op_num ; i++) {
                    //batch_size_list[scatter_op_index]
                    allocation_sizes[i] = num_merge_tiles * batch_size_list[i] * sizeof(KeyValuePairT);       // bytes needed for block carry-out pairs
                }
                allocation_sizes[scatter_op_num] = scatter_op_num * sizeof(KeyValuePairT *);       // bytes needed for block carry-out pairs

                if (CubDebug(error = ScanTileStateT::AllocationSize(num_segment_fixup_tiles, allocation_sizes[scatter_op_num + 1]))) break;    // bytes needed for reduce-by-key tile status descriptors
                allocation_sizes[scatter_op_num + 2] = (num_merge_tiles + 1) * sizeof(CoordinateT);   // bytes needed for tile starting coordinates

                // Alias the temporary allocations from the single storage blob (or compute the necessary size of the blob)
                void* allocations[scatter_op_num + 3] = {};
                if (CubDebug(error = AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes))) break;
                if (d_temp_storage == NULL)
                {
                    // Return if the caller is simply requesting the size of the storage allocation
                    break;
                }

                // Construct the tile status interface
                KeyValuePairT * carry_out_pairs_list[scatter_op_num];
                for (int i = 0; i < scatter_op_num ; i++) {
                    carry_out_pairs_list[i] = (KeyValuePairT *)allocations[i];
                }

                CubDebugExit(cudaMemcpy((void *)allocations[scatter_op_num], (const void *)carry_out_pairs_list, scatter_op_num * sizeof(KeyValuePairT *), cudaMemcpyHostToDevice));
                ScanTileStateT tile_state;
                if (CubDebug(error = tile_state.Init(num_segment_fixup_tiles, allocations[scatter_op_num + 1], allocation_sizes[scatter_op_num + 1]))) break;

                // Alias the other allocations
                KeyValuePairT**  d_tile_carry_pairs_list      = (KeyValuePairT**) allocations[scatter_op_num];  // Agent carry-out pairs
                CoordinateT*    d_tile_coordinates      = (CoordinateT*) allocations[scatter_op_num + 2];    // Agent starting coordinates
            #endif


            // Get search/init grid dims
            int search_block_size   = INIT_KERNEL_THREADS;
            int search_grid_size    = cub::DivideAndRoundUp(num_merge_tiles + 1, search_block_size);

            if (search_grid_size < sm_count)
            {
                // Not enough spmv tiles to saturate the device: have spmv blocks search their own staring coords
                d_tile_coordinates = NULL;
            }
            else
            {
                // Use separate search kernel if we have enough spmv tiles to saturate the device

                // Log spmv_search_kernel configuration
                #ifdef CUB_DETAIL_DEBUG_ENABLE_LOG
                _CubLog("Invoking spmv_search_kernel<<<%d, %d, 0, %lld>>>()\n",
                    search_grid_size, search_block_size, (long long) stream);
                #endif

                // Invoke spmv_search_kernel
                THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(
                    search_grid_size, search_block_size, 0, stream
                ).doit(spmv_search_kernel,
                    num_merge_tiles,
                    d_tile_coordinates,
                    easier_params);

                // Check for failure to launch
                if (CubDebug(error = cudaPeekAtLastError())) break;

                // Sync the stream if specified to flush runtime errors
                error = detail::DebugSyncStream(stream);
                if (CubDebug(error))
                {
                  break;
                }
            }

            // Log spmv_kernel configuration
            #ifdef CUB_DETAIL_DEBUG_ENABLE_LOG
            _CubLog("Invoking spmv_kernel<<<{%d,%d,%d}, %d, 0, %lld>>>(), %d items per thread, %d SM occupancy\n",
                spmv_grid_size.x, spmv_grid_size.y, spmv_grid_size.z, spmv_config.block_threads, (long long) stream, spmv_config.items_per_thread, spmv_sm_occupancy);
            #endif

            // Invoke spmv_kernel
            THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(
                spmv_grid_size, spmv_config.block_threads, 0, stream
            ).doit(spmv_kernel,
                easier_params,
                d_tile_coordinates,
                // d_tile_carry_pairs,//output
                d_tile_carry_pairs_list,//output
                // (KeyValuePairT *)allocations[0],//output
                num_merge_tiles,
                #ifdef USE_MULTI_STREAM_PPOST
                (void *)tile_state_list,
                #else
                tile_state,
                #endif
                0,//batch size, not used here
                num_segment_fixup_tiles);

            // Check for failure to launch
            if (CubDebug(error = cudaPeekAtLastError())) break;

            // Sync the stream if specified to flush runtime errors
            error = detail::DebugSyncStream(stream);
            if (CubDebug(error))
            {
              break;
            }

            // Run reduce-by-key fixup if necessary
            if (num_merge_tiles > 1)
            {
                #ifdef USE_MULTI_STREAM_PPOST
                //Stream的初始化
                cudaStream_t streams[BATCH_SIZE];
                for (int i = 0; i < BATCH_SIZE; i++) {
                    cudaStreamCreate(&streams[i]);
                }
                // for (int batch_idx=BatchSize-1; batch_idx >= 0; --batch_idx) {
                for (int batch_idx=0; batch_idx < BatchSize; ++batch_idx) {
                    // Log segment_fixup_kernel configuration
                    #ifdef CUB_DETAIL_DEBUG_ENABLE_LOG
                    _CubLog("Invoking segment_fixup_kernel<<<{%d,%d,%d}, %d, 0, %lld>>>(), %d items per thread, %d SM occupancy\n",
                        segment_fixup_grid_size.x, segment_fixup_grid_size.y, segment_fixup_grid_size.z, segment_fixup_config.block_threads, (long long) stream, segment_fixup_config.items_per_thread, segment_fixup_sm_occupancy);
                    #endif
                    // bool is_last = (batch_idx == 0) ? true: false;
                    // Invoke segment_fixup_kernel
                    THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(
                        segment_fixup_grid_size, segment_fixup_config.block_threads,
                        0, streams[batch_idx]
                    ).doit(segment_fixup_kernel,
                        d_tile_carry_pairs + num_merge_tiles * batch_idx,
                        easier_params.d_vector_y + easier_params.num_rows * batch_idx,
                        num_merge_tiles,
                        num_segment_fixup_tiles,
                        tile_state[batch_idx]);

                    // Check for failure to launch
                    if (CubDebug(error = cudaPeekAtLastError())) break;
                }

                    // Sync the stream if specified to flush runtime errors
                for (int batch_idx=0; batch_idx < BatchSize; ++batch_idx) {
                    error = detail::DebugSyncStream(streams[batch_idx]);
                    if (CubDebug(error))
                    {
                    break;
                    }
                }
                #else
                for (int scatter_op_index = 0; scatter_op_index < scatter_op_num; scatter_op_index++) {
                    for (int batch_idx=0; batch_idx < batch_size_list[scatter_op_index]; ++batch_idx) {
                        // Log segment_fixup_kernel configuration
                        #ifdef CUB_DETAIL_DEBUG_ENABLE_LOG
                        _CubLog("Invoking segment_fixup_kernel<<<{%d,%d,%d}, %d, 0, %lld>>>(), %d items per thread, %d SM occupancy\n",
                            segment_fixup_grid_size.x, segment_fixup_grid_size.y, segment_fixup_grid_size.z, segment_fixup_config.block_threads, (long long) stream, segment_fixup_config.items_per_thread, segment_fixup_sm_occupancy);
                        #endif
                        // bool is_last = (batch_idx == batch_size_list[scatter_op_index] - 1) ? true: false;
                        bool is_last = false;
                        // Invoke segment_fixup_kernel
                        THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(
                            segment_fixup_grid_size, segment_fixup_config.block_threads,
                            0, stream
                        ).doit(segment_fixup_kernel,
                            carry_out_pairs_list[scatter_op_index] + num_merge_tiles * batch_idx,
                            result_addr_list[scatter_op_index] + easier_params.num_rows * batch_idx,
                            num_merge_tiles,
                            num_segment_fixup_tiles,
                            tile_state);

                        // Check for failure to launch
                        error = detail::DebugSyncStream(stream);
                        if (CubDebug(error = cudaPeekAtLastError())) break;

                        // Sync the stream if specified to flush runtime errors
                        if (!is_last) {
                            #ifdef CUB_DETAIL_DEBUG_ENABLE_LOG
                            _CubLog("Invoking init_tile_state_kernel<<<{%d,%d,%d}, %d, 0, %lld>>>(), %d items per thread, %d SM occupancy\n",
                                tile_init_state_grid_size.x, tile_init_state_grid_size.y, tile_init_state_grid_size.z, tile_state_init_config.block_threads, (long long) stream, tile_state_init_config.items_per_thread, tile_state_init_sm_occupancy);
                            #endif
                            THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(
                                tile_init_state_grid_size, tile_state_init_config.block_threads,
                                0, stream
                            ).doit(tile_state_init_kernel,
                                num_segment_fixup_tiles,
                                tile_state);
                            error = detail::DebugSyncStream(stream);
                            if (CubDebug(error = cudaPeekAtLastError())) break;
                        }
                    }
                }
                if (CubDebug(error))
                    break;
                #endif



            }
        }
        while (0);

        return error;
    }

    /**
     * Internal dispatch routine for computing a device-wide reduction
     */
    CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t Dispatch(
        void*                   d_temp_storage,                     ///< [in] Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t&                 temp_storage_bytes,                 ///< [in,out] Reference to size in bytes of \p d_temp_storage allocation
        EasierParamsT&            easier_params,                        ///< SpMV input parameter bundle
        cudaStream_t            stream = 0)                         ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
    {
        cudaError error = cudaSuccess;
        do
        {
            // Get PTX version
            int ptx_version = 0;
            if (CubDebug(error = PtxVersion(ptx_version))) break;

            // Get kernel kernel dispatch configurations
            KernelConfig spmv_config, segment_fixup_config, tile_state_init_config;
            InitConfigs(ptx_version, spmv_config, segment_fixup_config, tile_state_init_config);

            constexpr bool has_alpha = false;
            constexpr bool has_beta = false;

            if (CubDebug(error = Dispatch(
                d_temp_storage, temp_storage_bytes, easier_params, stream, 
                MyDeviceSpmvSearchKernel<PtxSpmvPolicyT, OffsetT, CoordinateT, EasierParamsT>,
                MyDeviceSpmvKernel<PtxSpmvPolicyT, ScanTileStateT, ValueT, OffsetT, CoordinateT, has_alpha, has_beta>,
                MyDeviceSegmentFixupKernel<PtxSegmentFixupPolicy, KeyValuePairT*, ValueT*, OffsetT, ScanTileStateT>,
                MyDeviceTileStateInitKernel<PtxTileStateInitPolicy, ScanTileStateT>,
                spmv_config, segment_fixup_config, tile_state_init_config))) break;

        }
        while (0);

        return error;
    }

};


CUB_NAMESPACE_END


