#include <assert.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "gtest/gtest.h"

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>

#include <cooperative_groups.h>

#include "helper.h"

__global__ void baseline_kernel(const int* d_in, int* d_out, const int size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    assert(tid < size);
    d_out[tid] = d_in[size - 1 - tid];
}

__global__ void cg_sync_kernel(int* d_in, const int size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    assert(tid < size);

    // read
    int in = d_in[tid];

    // sync
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    grid.sync();

    // write
    d_in[size - 1 - tid] = in;
}

bool verify(thrust::device_vector<int>& d_out, thrust::host_vector<int>& h_out)
{
    thrust::copy(d_out.begin(), d_out.end(), h_out.begin());
    CUDA_ERROR(cudaDeviceSynchronize());

    for (int i = 0; i < h_out.size(); ++i) {
        if ((h_out[i] != h_out.size() - 1 - i)) {
            return false;
        }
    }
    return true;
}

float run_baseline(int                         arr_size,
                   thrust::device_vector<int>& d_in,
                   thrust::device_vector<int>& d_out,
                   dim3                        dimBlock,
                   dim3                        dimGrid,
                   size_t                      dynamicSMemSize,
                   cudaStream_t                stream,
                   int                         num_runs)
{
    int*  d_in_ptr     = d_in.data().get();
    int*  d_out_ptr    = d_out.data().get();
    void* kernelArgs[] = {&d_in_ptr, &d_out_ptr, &arr_size};
    CUDA_ERROR(cudaDeviceSynchronize());
    CUDATimer timer;
    timer.start();
    for (int d = 0; d < num_runs; ++d) {
        CUDA_ERROR(cudaLaunchCooperativeKernel((void*)baseline_kernel,
                                               dimGrid,
                                               dimBlock,
                                               kernelArgs,
                                               dynamicSMemSize,
                                               stream));
    }
    timer.stop();

    CUDA_ERROR(cudaDeviceSynchronize());

    return timer.elapsed_millis();
}


float run_cg_sync(int                         arr_size,
                  thrust::device_vector<int>& d_in,
                  dim3                        dimBlock,
                  dim3                        dimGrid,
                  size_t                      dynamicSMemSize,
                  cudaStream_t                stream,
                  int                         num_runs)
{
    int*  d_in_ptr     = d_in.data().get();
    void* kernelArgs[] = {&d_in_ptr, &arr_size};
    CUDA_ERROR(cudaDeviceSynchronize());
    CUDATimer timer;
    timer.start();
    for (int d = 0; d < num_runs; ++d) {
        CUDA_ERROR(cudaLaunchCooperativeKernel((void*)cg_sync_kernel,
                                               dimGrid,
                                               dimBlock,
                                               kernelArgs,
                                               dynamicSMemSize,
                                               stream));
    }
    timer.stop();

    CUDA_ERROR(cudaDeviceSynchronize());

    return timer.elapsed_millis();
}


TEST(Test, SingleRound)
{
    int dev                = 0;
    int supportsCoopLaunch = 0;
    CUDA_ERROR(cudaDeviceGetAttribute(
        &supportsCoopLaunch, cudaDevAttrCooperativeLaunch, dev));


    auto printt = [](auto d, bool end = false) {
        std::cout.fill(' ');
        std::cout << std::left << std::setw(20) << std::setfill(' ') << d;
        if (end) {
            std::cout << std::endl;
        }
    };

    printt("Threads");
    printt("Blocks");
    printt("Size");
    printt("Baseline (ms)");
    printt("CGSync (ms)", true);


    std::vector<int> threads{32, 64, 128, 256, 512, 1024};
    for (auto numThreads : threads) {

        int          numBlocksPerSm  = 0;
        size_t       dynamicSMemSize = 0;
        cudaStream_t stream          = NULL;
        int          arr_size        = 0;
        int          num_runs        = 1001;

        if (supportsCoopLaunch != 0) {
            cudaDeviceProp deviceProp;
            CUDA_ERROR(cudaGetDeviceProperties(&deviceProp, dev));
            CUDA_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &numBlocksPerSm, cg_sync_kernel, numThreads, dynamicSMemSize));

            dim3 dimBlock(numThreads, 1, 1);
            dim3 dimGrid(deviceProp.multiProcessorCount * numBlocksPerSm, 1, 1);

            arr_size = dimGrid.x * dimBlock.x;
            thrust::device_vector<int> d_in(arr_size);
            thrust::device_vector<int> d_out(arr_size);

            thrust::host_vector<int> h_in(arr_size);
            thrust::host_vector<int> h_out(arr_size);

            thrust::sequence(h_in.begin(), h_in.end());
            thrust::copy(h_in.begin(), h_in.end(), d_in.begin());


            double baseline_ms = run_baseline(arr_size,
                                              d_in,
                                              d_out,
                                              dimBlock,
                                              dimGrid,
                                              dynamicSMemSize,
                                              stream,
                                              num_runs);
            EXPECT_TRUE(verify(d_out, h_out)) << " Baseline Failed";

            double cg_sync_ms = run_cg_sync(arr_size,
                                            d_in,
                                            dimBlock,
                                            dimGrid,
                                            dynamicSMemSize,
                                            stream,
                                            num_runs);
            EXPECT_TRUE(verify(d_in, h_out)) << " Baseline Failed";

            printt(dimBlock.x);
            printt(dimGrid.x);
            printt(arr_size);
            printt(baseline_ms);
            printt(cg_sync_ms, true);
        }
    }
}


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
