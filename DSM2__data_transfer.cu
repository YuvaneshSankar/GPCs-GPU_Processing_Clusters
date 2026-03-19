#include <cuda_runtime.h>
#include <stdio.h>
#include <cooperative_groups.h>

#define CUDA_CHECK(err) do { \
    if (err != cudaSuccess) { \
        printf("CUDA error: %s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

__global__ void dsm_kernel(float* final_result, int num_itr) {
    auto cluster = cooperative_groups::this_cluster();
    int rank = cluster.block_rank();
    int idx = threadIdx.x;

    extern __shared__ float smem[];

    // Initial load from global to producer shared
    if (rank == 0) {
        for (int j = idx; j < 4096; j += blockDim.x) {
            smem[j] = final_result[j];
        }
    }
    cluster.sync();

    // Repeated handoff
    for (int i = 0; i < num_itr; i++) {
        if (rank == 0) {
            for (int j = idx; j < 4096; j += blockDim.x) {
                smem[j] += 1.0f;
            }
        }

        cluster.sync();

        if (rank == 1) {
            float* producer_smem = cluster.map_shared_rank(smem, 0);
            for (int j = idx; j < 4096; j += blockDim.x) {
                producer_smem[j] += 2.0f;
            }
        }

        cluster.sync();
    }

    // Final copy back from producer shared to global
    if (rank == 1) {
        float* producer_smem = cluster.map_shared_rank(smem, 0);
        for (int j = idx; j < 4096; j += blockDim.x) {
            final_result[j] = producer_smem[j];
        }
    }
}

int main() {
    const int N = 4096;
    float* h_result = (float*)malloc(N * sizeof(float));
    float* d_result = nullptr;

    for (int i = 0; i < N; i++) h_result[i] = 1.0f;

    CUDA_CHECK(cudaMalloc(&d_result, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_result, h_result, N * sizeof(float), cudaMemcpyHostToDevice));

    size_t shared_bytes = N * sizeof(float);

    dim3 block(256, 1, 1);
    dim3 grid(8, 1, 1);  // 8 blocks = 4 clusters of 2

    int num_itr = 10000;

    // Print device info
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("Device: %s, SMs: %d, Shared mem/block: %zu KB\n", prop.name, prop.multiProcessorCount, prop.sharedMemPerBlock / 1024);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    // Runtime cluster configuration (more reliable)
    cudaLaunchConfig_t config = {0};
    config.gridDim = grid;
    config.blockDim = block;
    config.sharedMemBytes = shared_bytes;

    cudaLaunchAttribute attr;
    attr.attr = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim.x = 2;
    attr.val.clusterDim.y = 1;
    attr.val.clusterDim.z = 1;
    config.attrs = &attr;
    config.numAttrs = 1;

    CUDA_CHECK(cudaLaunchKernelEx(&config, dsm_kernel, d_result, num_itr));

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("DSM Total time: %.3f ms\n", ms);
    printf("DSM Per handoff: %.3f us\n", ms * 1000.0f / num_itr);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_result, d_result, N * sizeof(float), cudaMemcpyDeviceToHost));

    printf("Final value at index 0: %f (expected ~ %.0f)\n", h_result[0], 1.0f + 3.0f * num_itr);

    free(h_result);
    CUDA_CHECK(cudaFree(d_result));

    return 0;
}