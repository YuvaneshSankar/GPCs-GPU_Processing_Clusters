#include <cuda_runtime.h>
#include <stdio.h>
#include <cooperative_groups.h>

#define CUDA_CHECK(err) do { \
    if (err != cudaSuccess) { \
        printf("CUDA error: %s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

__global__ void __cluster_dims__(2, 1, 1) dsm_kernel(float* final_result, int num_itr) {
    auto cluster = cooperative_groups::this_cluster();
    int rank = cluster.block_rank();
    int idx = threadIdx.x;

    extern __shared__ float smem[];

    if (idx != 0) return;


    for (int i = 0; i < num_itr; i++) {
        if (rank == 0) {
          float value = final_result[idx];
          smem[idx] = value+1.0f;
        }

        cluster.sync();

        if (rank == 1) {
            float* producer_smem = cluster.map_shared_rank(smem, 0);
            float current = producer_smem[idx];
            current += 2.0f;
            producer_smem[idx] = current;
        }

        cluster.sync();
    }

    // Only consumer writes final value back to global memory
    if (rank == 1) {
        final_result[0] = smem[idx];
    }
}

void init_matrix(float* in) {
    for (int i = 0; i < 4096; i++) {
        in[i] = 1.0f;
    }
}

int main() {
    float* h_result = (float*)malloc(sizeof(float));
    float* d_result = nullptr;

    *h_result = 1.0f;

    CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_result, h_result, sizeof(float), cudaMemcpyHostToDevice));

    size_t shared_bytes = sizeof(float);

    dim3 block(32, 1, 1);
    dim3 grid(4, 1, 1);

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

    CUDA_CHECK(cudaMemcpy(h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost));

    printf("Final value: %f (expected ~ %.0f)\n", *h_result, 1.0f + 3.0f * num_itr);

    free(h_result);
    CUDA_CHECK(cudaFree(d_result));

    return 0;
}