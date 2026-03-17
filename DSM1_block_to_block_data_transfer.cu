#include <cuda_runtime.h>
#include <stdio.h>
#include <cooperative_groups.h>

#define CUDA_CHECK(err) do { \
    if (err != cudaSuccess) { \
        printf("CUDA error: %s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

__global__ void global_memory_kernel(float* a, int num_itr) {
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    int idx = threadIdx.x;

    if (idx >= 4096) return;

    for (int i = 0; i < num_itr; i++) {
        if (blockIdx.x == 0) {
            a[idx] += 1.0f;
        }
        grid.sync();
        if (blockIdx.x == 1) {
            a[idx] += 2.0f;
        }
        grid.sync();
    }
}

void init_matrix(float* in) {
    for (int i = 0; i < 4096; i++) {
        in[i] = 1.0f;
    }
}

int main() {
    float* h_array = nullptr;
    float* d_array = nullptr;
    size_t tile_size = 4096 * sizeof(float);

    h_array = (float*)malloc(tile_size);
    if (!h_array) {
        printf("Host malloc failed\n");
        return 1;
    }
    init_matrix(h_array);

    CUDA_CHECK(cudaMalloc((void**)&d_array, tile_size));
    CUDA_CHECK(cudaMemcpy(d_array, h_array, tile_size, cudaMemcpyHostToDevice));

    int major = 0, minor = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0);
    printf("Detected GPU compute capability: %d.%d\n", major, minor);

    dim3 block_size(1024, 1, 1);
    dim3 grid_size(2, 1, 1);
    int num_itr = 10000;

    void* kernel_args[] = { (void*)&d_array, (void*)&num_itr };

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    CUDA_CHECK(cudaLaunchCooperativeKernel(
        (void*)global_memory_kernel,
        grid_size,
        block_size,
        kernel_args,
        0,    // shared mem
        0     // stream
    ));

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Total kernel time: %.3f ms\n", ms);
    printf("Approximate time per handoff: %.3f us\n", (ms * 1000.0f) / num_itr);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_array, d_array, tile_size, cudaMemcpyDeviceToHost));

    printf("Final value at index 0: %f (expected ~ %.0f)\n", h_array[0], 1.0f + 3.0f * num_itr);

    free(h_array);
    CUDA_CHECK(cudaFree(d_array));

    return 0;
}