#include <cuda_runtime.h>
#include <stdio.h>
#include <cooperative_groups.h>

#define CUDA_CHECK(err) do { \
    if (err != cudaSuccess) { \
        printf("CUDA error: %s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

__global__ void __cluster_dims__(2, 1, 1)
dsm_kernel(float* final_result, int num_itr) {
    auto cluster = cooperative_groups::this_cluster();
    int rank = cluster.block_rank();
    int idx = threadIdx.x;

    extern __shared__ float smem[];   // producer's shared memory

    // Initial load from global to producer shared (once)
    if (rank == 0) {
        for (int j = idx; j < 4096; j += blockDim.x) {
            smem[j] = final_result[j];
        }
    }
    cluster.sync();

    // Repeat handoff num_itr times
    for (int i = 0; i < num_itr; i++) {
        if (rank == 0) {  // producer
            for (int j = idx; j < 4096; j += blockDim.x) {
                smem[j] += 1.0f;  // add +1 directly to shared
            }
        }

        cluster.sync();  // make visible to consumer

        if (rank == 1) {  // consumer
            float* producer_smem = cluster.map_shared_rank(smem, 0);
            for (int j = idx; j < 4096; j += blockDim.x) {
                producer_smem[j] += 2.0f;  // add +2 directly to producer's shared
            }
        }

        cluster.sync();  // ready for next iteration
    }

    // Final copy: consumer copies from producer's shared to global
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
    dim3 grid(8, 1, 1);  // 8 blocks = 4 clusters of 2 (larger to avoid launch failure)

    int num_itr = 10000;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    dsm_kernel<<<grid, block, shared_bytes>>>(d_result, num_itr);

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