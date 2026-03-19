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

    extern __shared__ float smem[];   // producer block's shared memory

    // Step 1: Producer copies the initial global array to shared memory (once)
    if (rank == 0) {
        for (int j = idx; j < 4096; j += blockDim.x) {
            smem[j] = final_result[j];
        }
    }
    cluster.sync();

    // Step 2: Repeat the handoff 10,000 times
    for (int i = 0; i < num_itr; i++) {
        if (rank == 0) {  // producer block
            for (int j = idx; j < 4096; j += blockDim.x) {
                smem[j] += 1.0f;          // add +1 to its own shared memory
            }
        }

        cluster.sync();   // make producer's writes visible to consumer

        if (rank == 1) {  // consumer block
            float* producer_smem = cluster.map_shared_rank(smem, 0);
            for (int j = idx; j < 4096; j += blockDim.x) {
                producer_smem[j] += 2.0f;  // add +2 directly to producer's shared memory
            }
        }

        cluster.sync();   // wait for consumer to finish
    }

    // Step 3: Consumer copies final result back to global memory (once)
    if (rank == 1) {
        for (int j = idx; j < 4096; j += blockDim.x) {
            final_result[j] = smem[j];   // read from producer's shared via map
        }
    }
}

int main() {
    float* h_result = (float*)malloc(4096 * sizeof(float));
    float* d_result = nullptr;

    for (int i = 0; i < 4096; i++) h_result[i] = 1.0f;

    CUDA_CHECK(cudaMalloc(&d_result, 4096 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_result, h_result, 4096 * sizeof(float), cudaMemcpyHostToDevice));

    size_t shared_bytes = 4096 * sizeof(float);

    dim3 block(256, 1, 1);
    dim3 grid(4, 1, 1);   // must be multiple of cluster size 2

    int num_itr = 10000;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    dsm_kernel<<<grid, block, shared_bytes>>>(d_result, num_itr);

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("DSM Total time: %.3f ms\n", ms);
    printf("DSM Per handoff: %.3f us\n", ms * 1000.0f / num_itr);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaMemcpy(h_result, d_result, 4096 * sizeof(float), cudaMemcpyDeviceToHost));

    printf("Final value at index 0: %f (expected ~ %.0f)\n", h_result[0], 1.0f + 3.0f * num_itr);

    free(h_result);
    CUDA_CHECK(cudaFree(d_result));

    return 0;
}