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
    int in = d_in[tid];

    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    grid.sync();

    d_in[size - 1 - tid] = in;
}

TEST(Test, exe)
{
    int dev                = 0;
    int supportsCoopLaunch = 0;
    CUDA_ERROR(cudaDeviceGetAttribute(
        &supportsCoopLaunch, cudaDevAttrCooperativeLaunch, dev));

    int          numBlocksPerSm  = 0;
    const int    numThreads      = 256;
    size_t       dynamicSMemSize = 0;
    cudaStream_t stream          = NULL;
    int          arr_size        = 0;

    if (supportsCoopLaunch != 0) {
        cudaDeviceProp deviceProp;
        CUDA_ERROR(cudaGetDeviceProperties(&deviceProp, dev));
        CUDA_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &numBlocksPerSm, baseline_kernel, numThreads, dynamicSMemSize));

        dim3 dimBlock(numThreads, 1, 1);
        dim3 dimGrid(deviceProp.multiProcessorCount * numBlocksPerSm, 1, 1);

        arr_size = dimGrid.x * dimBlock.x;
        thrust::device_vector<int> d_in(arr_size);
        thrust::device_vector<int> d_out(arr_size);

        thrust::host_vector<int> h_in(arr_size);
        thrust::sequence(h_in.begin(), h_in.end());


        thrust::copy(h_in.begin(), h_in.end(), d_in.begin());

        int* d_in_ptr  = d_in.data().get();
        int* d_out_ptr = d_out.data().get();

        void* kernelArgs[] = {&d_in_ptr, &d_out_ptr, &arr_size};

        CUDA_ERROR(cudaDeviceSynchronize());

        CUDATimer timer;
        timer.start();
        for (int d = 0; d < 1000; ++d) {
            CUDA_ERROR(cudaLaunchCooperativeKernel((void*)baseline_kernel,
                                                   dimGrid,
                                                   dimBlock,
                                                   kernelArgs,
                                                   dynamicSMemSize,
                                                   stream));
        }
        timer.stop();

        CUDA_ERROR(cudaDeviceSynchronize());

        thrust::host_vector<int> h_out(arr_size);
        thrust::copy(d_out.begin(), d_out.end(), h_out.begin());
        CUDA_ERROR(cudaDeviceSynchronize());

        for (int i = 0; i < h_out.size(); ++i) {
            EXPECT_EQ(h_out[i], arr_size - 1 - i);
        }

        std::cout << "Baseline took= " << timer.elapsed_millis() << " (ms)"
                  << std::endl;
    }
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
