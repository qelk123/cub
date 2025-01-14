/******************************************************************************
 * Copyright (c) 2011-2016, NVIDIA CORPORATION.  All rights reserved.
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
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIAeBILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

//---------------------------------------------------------------------
// SpMV comparison tool
//---------------------------------------------------------------------

#include <stdio.h>
#include <map>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <fstream>

#include <cusparse.h>

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR
#ifndef BLOCK_SIZE
    #define BLOCK_SIZE 64
#endif

#ifndef ITEM_PER_THREAD
    #define ITEM_PER_THREAD 3
#endif
const int BATCH_SIZE = BATCH_SIZE_D;

#include <cub/device/device_spmv.cuh>
#include <cub/auto_codegen_support/patch/device_spmv_patch.cuh>
#include <cub/util_allocator.cuh>
#include <cub/iterator/tex_ref_input_iterator.cuh>
#include "/home/v-yinuoliu/yinuoliu/code/thrust/examples/include/timer.h"

#include "sparse_matrix.h"
#include <utils.h>
#include <iostream>
#include <fstream>


using namespace cub;

#define CUB_DEBUG_ALL



//---------------------------------------------------------------------
// Globals, constants, and type declarations
//---------------------------------------------------------------------

bool                    g_quiet     = false;        // Whether to display stats in CSV format
bool                    g_verbose   = false;        // Whether to display output to console
bool                    g_verbose2  = false;        // Whether to display input to console
CachingDeviceAllocator  g_allocator(true);          // Caching allocator for device memory


//---------------------------------------------------------------------
// SpMV verification
//---------------------------------------------------------------------

// Compute reference SpMV y = Ax
template <
    typename ValueT,
    typename OffsetT>
void SpmvGold(
    CsrMatrix<ValueT, OffsetT>&     a,
    ValueT*                         vector_x,
    ValueT*                         vector_y_in,
    ValueT*                         vector_y_out,
    ValueT*                         vector_y_out_2,
    ValueT*                         batch_sparse_matrix,
    ValueT                          alpha,
    ValueT                          beta)
{
    // int BATCH_SIZE_INNER = 3;
    // for(int batch_idx = 0; batch_idx < BATCH_SIZE/BATCH_SIZE_INNER; batch_idx++) {
    //     for (OffsetT row = 0; row < a.num_rows; ++row)
    //     {
    //         ValueT partial = beta * vector_y_in[batch_idx * a.num_rows + row];
    //         for (
    //             OffsetT offset = a.row_offsets[row];
    //             offset < a.row_offsets[row + 1];
    //             ++offset)
    //         {
    //             for(int batch_idx_inner = 0; batch_idx_inner < BATCH_SIZE_INNER; batch_idx_inner++) {
    //                 partial += alpha * batch_sparse_matrix[offset + (batch_idx * BATCH_SIZE_INNER + batch_idx_inner) * a.num_nonzeros] * vector_x[a.column_indices[offset] + (batch_idx * BATCH_SIZE_INNER + batch_idx_inner) * a.num_rows];
    //             }
    //         }
    //         vector_y_out[batch_idx * a.num_rows + row] = partial;
    //     }
    // }

    // for(int batch_idx = 0; batch_idx < BATCH_SIZE; batch_idx++) {
    //     for (OffsetT row = 0; row < a.num_rows; ++row)
    //     {
    //         ValueT partial = beta * vector_y_in[batch_idx * a.num_rows + row];
    //         for (
    //             OffsetT offset = a.row_offsets[row];
    //             offset < a.row_offsets[row + 1];
    //             ++offset)
    //         {
    //             for(int batch_idx_inner = 0; batch_idx_inner < 1; batch_idx_inner++) {
    //                 partial += alpha * batch_sparse_matrix[offset + (batch_idx * 1 + batch_idx_inner) * a.num_nonzeros] * vector_x[a.column_indices[offset] + (batch_idx * 1 + batch_idx_inner) * a.num_rows];
    //                 // partial += alpha * batch_sparse_matrix[offset * BATCH_SIZE + (batch_idx * 1 + batch_idx_inner)] * vector_x[a.column_indices[offset] + (batch_idx * 1 + batch_idx_inner) * a.num_rows];
    //             }
    //         }
    //         vector_y_out[batch_idx * a.num_rows + row] = partial;
    //     }
    // }

    for(int batch_idx = 0; batch_idx < BATCH_SIZE; batch_idx++) {
            for (OffsetT row = 0; row < a.num_rows; ++row)
            {
                ValueT partial = beta * vector_y_in[batch_idx * a.num_rows + row];
                for (
                    OffsetT offset = a.row_offsets[row];
                    offset < a.row_offsets[row + 1];
                    ++offset)
                {
                    for(int batch_idx_inner = 0; batch_idx_inner < 3; batch_idx_inner++) {
                        partial += alpha * batch_sparse_matrix[offset + (batch_idx * 3 + batch_idx_inner) * a.num_nonzeros] * vector_x[a.column_indices[offset] + (batch_idx_inner) * a.num_rows];
                        // partial += alpha * batch_sparse_matrix[offset + (batch_idx * BATCH_SIZE + batch_idx_inner) * a.num_nonzeros] * vector_x[a.column_indices[offset]  * BATCH_SIZE + (batch_idx_inner)];
                        // partial += alpha * batch_sparse_matrix[offset * BATCH_SIZE + (batch_idx * 1 + batch_idx_inner)] * vector_x[a.column_indices[offset] + (batch_idx * 1 + batch_idx_inner) * a.num_rows];
                    }
                }
                vector_y_out[batch_idx * a.num_rows + row] = partial;
            }
        }
    for(int batch_idx = 0; batch_idx < 12; batch_idx++) {
            for (OffsetT row = 0; row < a.num_rows; ++row)
            {
                ValueT partial = 0;
                for (
                    OffsetT offset = a.row_offsets[row];
                    offset < a.row_offsets[row + 1];
                    ++offset)
                {
                    for(int batch_idx_inner = 0; batch_idx_inner < 1; batch_idx_inner++) {
                        partial += batch_sparse_matrix[offset + (batch_idx * 1 + batch_idx_inner) * a.num_nonzeros] * 2;
                        // partial += alpha * batch_sparse_matrix[offset + (batch_idx * BATCH_SIZE + batch_idx_inner) * a.num_nonzeros] * vector_x[a.column_indices[offset]  * BATCH_SIZE + (batch_idx_inner)];
                        // partial += alpha * batch_sparse_matrix[offset * BATCH_SIZE + (batch_idx * 1 + batch_idx_inner)] * vector_x[a.column_indices[offset] + (batch_idx * 1 + batch_idx_inner) * a.num_rows];
                    }
                }
                vector_y_out_2[batch_idx * a.num_rows + row] = partial;
            }
        }
}


/**
 * Run CUB SpMV
 */
template <
    int      BATCH_SIZE,
    typename ValueT,
    typename OffsetT>
float TestGpuMergeCsrmv(
    ValueT*                         vector_y_in,
    ValueT*                         reference_vector_y_out,
    ValueT*                         reference_vector_y_out_2,
    EasierParams<ValueT, OffsetT>&  params,
    int                             timing_iterations,
    float                           &setup_ms)
{
    setup_ms = 0.0;

    // Allocate temporary storage
    size_t temp_storage_bytes = 0;
    void *d_temp_storage = NULL;

    // Get amount of temporary storage needed
    CubDebugExit(Easier_Struct::Easier<BATCH_SIZE>(
        d_temp_storage, temp_storage_bytes, params,
        (cudaStream_t) 0, false));

    // Allocate
    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

    // Reset input/output vector y
    CubDebugExit(cudaMemcpy(params.d_vector_y_0, vector_y_in, sizeof(ValueT) * params.num_rows * BATCH_SIZE, cudaMemcpyHostToDevice));
    // CubDebugExit(cudaMemcpy(params.d_vector_y_1, vector_y_in, sizeof(ValueT) * params.num_rows * BATCH_SIZE, cudaMemcpyHostToDevice));

    // Warmup
    CubDebugExit(Easier_Struct::Easier<BATCH_SIZE>(
        d_temp_storage, temp_storage_bytes, params,
        (cudaStream_t) 0, false));

    ValueT *h_data = (ValueT*) malloc(params.num_rows * BATCH_SIZE * sizeof(ValueT));
    ValueT *h_data_2 = (ValueT*) malloc(params.num_rows * 12 * sizeof(ValueT));

    // Copy data back
    cudaMemcpy(h_data,      params.d_vector_y_0, sizeof(ValueT) * params.num_rows * BATCH_SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_data_2,    params.d_vector_y_1, sizeof(ValueT) * params.num_rows * 12,         cudaMemcpyDeviceToHost);

    //----------------------------------------check result----------------------------------------
    int compare = 0;
    int i = 0 ;
    #ifdef SAVE_TO_FILE
    std::ofstream outfile("output1.txt");
    #endif
    for (; i < params.num_rows  ; i++) {
        for (int j =0; j < BATCH_SIZE; j++) {
            if(abs(h_data[j * params.num_rows + i ] - reference_vector_y_out[j * params.num_rows + i ]) > 1e-6) {
                compare++;
            }
            #ifdef SAVE_TO_FILE
            if (outfile.is_open()) {
                outfile << std::to_string(reference_vector_y_out[j * params.num_rows + i ]) << std::endl;
            } else {
                std::cout << "can't open file\n" << std::endl;
            }
            #endif
        }
    }
    #ifdef SAVE_TO_FILE
    outfile.close();
    #endif
    if(compare > 0) {
        std::cout<<"wrong number:"<<compare<<"/"<< params.num_rows * BATCH_SIZE <<"\n";
        printf("\nd_vector_y_0:FAIL\n"); fflush(stdout);
    }

    compare = 0;
    #ifdef SAVE_TO_FILE
    std::ofstream outfile2("output2.txt");
    #endif
    for (i = 0 ; i < params.num_rows  ; i++) {
        for (int j =0; j < 12; j++) {
            if(abs(h_data_2[j * params.num_rows + i ] - reference_vector_y_out_2[j * params.num_rows + i ]) > 1e-6) {
                compare++;
            }
            #ifdef SAVE_TO_FILE
            if (outfile2.is_open()) {
                outfile2 << std::to_string(reference_vector_y_out[j * params.num_rows + i ]) << std::endl;
            } else {
                std::cout << "无法打开文件。" << std::endl;
            }
            #endif
        }
    }
    #ifdef SAVE_TO_FILE
    outfile2.close();
    #endif
    if(compare > 0) {
        std::cout<<"wrong number:"<<compare<<"/"<< params.num_rows * 12 <<"\n";
        printf("\nd_vector_y_1:FAIL\n"); fflush(stdout);
    }

    //------------------------------------------measure kernel time-----------------------------------------

    timer t;
    float elapsed_ms = 0.0;
    for(int it = 0; it < timing_iterations; ++it)
    {
        CubDebugExit(Easier_Struct::Easier<BATCH_SIZE>(
            d_temp_storage, temp_storage_bytes, params,
            (cudaStream_t) 0, false));
    }
    cudaThreadSynchronize();
    elapsed_ms = t.elapsed();

    return elapsed_ms / timing_iterations;
}

//---------------------------------------------------------------------
// Test generation
//---------------------------------------------------------------------

/**
 * Display perf
 */
template <typename ValueT, typename OffsetT>
void DisplayPerf(
    float                           device_giga_bandwidth,
    double                          setup_ms,
    double                          avg_ms,
    CsrMatrix<ValueT, OffsetT>&     csr_matrix)
{
    double nz_throughput, effective_bandwidth;
    size_t total_bytes = (csr_matrix.num_nonzeros * (sizeof(ValueT) * 2 + sizeof(OffsetT))) +
        (csr_matrix.num_rows) * (sizeof(OffsetT) + sizeof(ValueT));

    nz_throughput       = double(csr_matrix.num_nonzeros*BATCH_SIZE) / avg_ms / 1.0e6;
    effective_bandwidth = double(total_bytes) / avg_ms / 1.0e6;
    printf(" %.4f\n",avg_ms);
    fflush(stdout);
}



/**
 * Run tests
 */
template <
    typename ValueT,
    typename OffsetT>
void RunTest(
    ValueT                      alpha,
    ValueT                      beta,
    CooMatrix<ValueT, OffsetT>& coo_matrix,
    int                         timing_iterations,
    CommandLineArgs&            args)
{
    // Adaptive timing iterations: run 16 billion nonzeros through
    if (timing_iterations == -1)
        timing_iterations = std::min(50000ull, std::max(100ull, ((16ull << 30) / coo_matrix.num_nonzeros)));

    // Convert to CSR
    CsrMatrix<ValueT, OffsetT> csr_matrix(coo_matrix);
    if (!args.CheckCmdLineFlag("csrmv"))
        coo_matrix.Clear();

    if (!g_quiet)
    {
        printf("\n");
        csr_matrix.DisplayHistogram();
        printf("\n");
        if (g_verbose2)
            csr_matrix.Display();
        printf("\n");
    }
    fflush(stdout);


    // Allocate input and output vectors

    ValueT* vector_x            = new ValueT[csr_matrix.num_cols * 3];
    ValueT* vector_y_in         = new ValueT[csr_matrix.num_rows * BATCH_SIZE];
    ValueT* vector_y_out        = new ValueT[csr_matrix.num_rows * BATCH_SIZE];
    ValueT* vector_y_out_2      = new ValueT[csr_matrix.num_rows * 12];
    // ValueT* batch_matrix_val    = new ValueT[csr_matrix.num_nonzeros * BATCH_SIZE];
    ValueT* batch_matrix_val    = new ValueT[csr_matrix.num_nonzeros * BATCH_SIZE * 3];

    std::srand(0);
    for (int col = 0; col < csr_matrix.num_cols * 3; ++col)
        vector_x[col] = (rand()%100)/10.0;
    for (int idx = 0; idx < csr_matrix.num_nonzeros ; ++idx)
        for (int batch_id = 0; batch_id < BATCH_SIZE; ++batch_id)
            for (int batch_id_2 = 0; batch_id_2 < 3; ++batch_id_2)
                batch_matrix_val[(batch_id * 3 + batch_id_2 )*csr_matrix.num_nonzeros + idx] = (rand()%100)/10.0;

    // Compute reference answer
    SpmvGold(csr_matrix, vector_x, vector_y_in, vector_y_out, vector_y_out_2, batch_matrix_val, alpha, beta);

    float avg_ms, setup_ms;

    // Get GPU device bandwidth (GB/s)
    float device_giga_bandwidth = args.device_giga_bandwidth;

    // Allocate and initialize GPU problem
    EasierParams<ValueT, OffsetT> params;
    
    params.num_rows         = csr_matrix.num_rows;
    params.num_cols         = csr_matrix.num_cols;
    params.num_nonzeros     = csr_matrix.num_nonzeros;
    params.alpha            = alpha;
    params.beta             = beta;

    CubDebugExit(g_allocator.DeviceAllocate((void **) &params.e1,           sizeof(ValueT) * csr_matrix.num_nonzeros * BATCH_SIZE * 3));
    CubDebugExit(g_allocator.DeviceAllocate((void **) &params.G_0,          sizeof(OffsetT) * csr_matrix.num_nonzeros));
    CubDebugExit(g_allocator.DeviceAllocate((void **) &params.v1,           sizeof(ValueT) * csr_matrix.num_cols * 3));
    const OffsetT*  d_row_offsets;
    CubDebugExit(g_allocator.DeviceAllocate((void **) &d_row_offsets,       sizeof(OffsetT) * (csr_matrix.num_rows + 1)));
    CubDebugExit(g_allocator.DeviceAllocate((void **) &params.d_vector_y_0, sizeof(ValueT) * csr_matrix.num_rows * BATCH_SIZE));
    CubDebugExit(g_allocator.DeviceAllocate((void **) &params.d_vector_y_1, sizeof(ValueT) * csr_matrix.num_rows * 12));
    params.d_row_end_offsets = d_row_offsets + 1;

    CubDebugExit(cudaMemcpy((void *)params.e1,       (const void *)batch_matrix_val,            sizeof(ValueT) * csr_matrix.num_nonzeros * BATCH_SIZE * 3, cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemcpy((void *)params.G_0,      (const void *)csr_matrix.column_indices,   sizeof(OffsetT) * csr_matrix.num_nonzeros, cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemcpy((void *)params.v1,       (const void *)vector_x,                    sizeof(ValueT) * csr_matrix.num_cols * 3, cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemcpy((void *)d_row_offsets,   (const void *)csr_matrix.row_offsets,      sizeof(OffsetT) * (csr_matrix.num_rows + 1), cudaMemcpyHostToDevice));

	// Merge-based
    avg_ms = TestGpuMergeCsrmv<BATCH_SIZE>(vector_y_in, vector_y_out, vector_y_out_2, params, timing_iterations, setup_ms);
    DisplayPerf(device_giga_bandwidth, setup_ms, avg_ms, csr_matrix);

}

/**
 * Run tests
 */
template <
    typename ValueT,
    typename OffsetT>
void RunTests(
    ValueT              alpha,
    ValueT              beta,
    const std::string&  mtx_filename,
    int                 grid2d,
    int                 grid3d,
    int                 wheel,
    int                 dense,
    int                 timing_iterations,
    CommandLineArgs&    args)
{
    // Initialize matrix in COO form
    CooMatrix<ValueT, OffsetT> coo_matrix;

    if (!mtx_filename.empty())
    {
        // Parse matrix market file
        coo_matrix.InitMarket(mtx_filename, 1.0, false);

        if ((coo_matrix.num_rows == 1) || (coo_matrix.num_cols == 1) || (coo_matrix.num_nonzeros == 1))
        {
            if (!g_quiet) printf("Trivial dataset\n");
            exit(0);
        }
        // printf("%s, ", mtx_filename.c_str()); fflush(stdout);
    }
    else if (grid2d > 0)
    {
        // Generate 2D lattice
        printf("grid2d_%d, ", grid2d); fflush(stdout);
        coo_matrix.InitGrid2d(grid2d, false);
    }
    else if (grid3d > 0)
    {
        // Generate 3D lattice
        printf("grid3d_%d, ", grid3d); fflush(stdout);
        coo_matrix.InitGrid3d(grid3d, false);
    }
    else if (wheel > 0)
    {
        // Generate wheel graph
        printf("wheel_%d, ", grid2d); fflush(stdout);
        coo_matrix.InitWheel(wheel);
    }
    else if (dense > 0)
    {
        // Generate dense graph
        OffsetT size = 1 << 24; // 16M nnz
        args.GetCmdLineArgument("size", size);

        OffsetT rows = size / dense;
        printf("dense_%d_x_%d, ", rows, dense); fflush(stdout);
        coo_matrix.InitDense(rows, dense);
    }
    else
    {
        fprintf(stderr, "No graph type specified.\n");
        exit(1);
    }

    RunTest(
        alpha,
        beta,
        coo_matrix,
        timing_iterations,
        args);
}

// sudo /usr/local/cuda-12.1/bin/ncu --set full -f --export ./merge_based_2388 ./gpu_spmv --mtx=/home/v-yinuoliu/yinuoliu/code/SparseCodegen/matrix2388.mtx --i=1
// ./gpu_spmv --mtx=/home/v-yinuoliu/yinuoliu/code/SparseCodegen/matrix2388.mtx --i=1

/**
 * Main
 */
int main(int argc, char **argv)
{
    // Initialize command line
    CommandLineArgs args(argc, argv);
    if (args.CheckCmdLineFlag("help"))
    {
        printf(
            "%s "
            "[--csrmv | --hybmv | --bsrmv ] "
            "[--device=<device-id>] "
            "[--quiet] "
            "[--v] "
            "[--i=<timing iterations>] "
            "[--fp32] "
            "[--alpha=<alpha scalar (default: 1.0)>] "
            "[--beta=<beta scalar (default: 0.0)>] "
            "\n\t"
                "--mtx=<matrix market file> "
            "\n\t"
                "--dense=<cols>"
            "\n\t"
                "--grid2d=<width>"
            "\n\t"
                "--grid3d=<width>"
            "\n\t"
                "--wheel=<spokes>"
            "\n", argv[0]);
        exit(0);
    }

    bool                fp32;
    std::string         mtx_filename;
    int                 grid2d              = -1;
    int                 grid3d              = -1;
    int                 wheel               = -1;
    int                 dense               = -1;
    int                 timing_iterations   = -1;
    float               alpha               = 1.0;
    float               beta                = 0.0;

    g_verbose = args.CheckCmdLineFlag("v");
    g_verbose2 = args.CheckCmdLineFlag("v2");
    g_quiet = args.CheckCmdLineFlag("quiet");
    fp32 = args.CheckCmdLineFlag("fp32");
    args.GetCmdLineArgument("i", timing_iterations);
    args.GetCmdLineArgument("mtx", mtx_filename);
    args.GetCmdLineArgument("grid2d", grid2d);
    args.GetCmdLineArgument("grid3d", grid3d);
    args.GetCmdLineArgument("wheel", wheel);
    args.GetCmdLineArgument("dense", dense);
    args.GetCmdLineArgument("alpha", alpha);
    args.GetCmdLineArgument("beta", beta);


    std::cout<<mtx_filename<<" "<<BATCH_SIZE<<" "<<BLOCK_SIZE<<" "<<ITEM_PER_THREAD;

    mtx_filename = "/home/v-yinuoliu/yinuoliu/code/SparseCodegen/matrix" + mtx_filename + ".mtx";
    // Initialize device
    CubDebugExit(args.DeviceInit());

    // Run test(s)
    if (fp32)
    {
        RunTests<float, int>(alpha, beta, mtx_filename, grid2d, grid3d, wheel, dense, timing_iterations, args);
    }
    else
    {
        RunTests<double, int>(alpha, beta, mtx_filename, grid2d, grid3d, wheel, dense, timing_iterations, args);
    }

    CubDebugExit(cudaDeviceSynchronize());
    printf("\n");

    return 0;
}
