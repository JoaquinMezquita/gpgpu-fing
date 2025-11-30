#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"

#define ARRAY_SIZE 8192
#define BLOCK_SIZE 1024
#define INITIAL_VALUE 0

#define CUDA_CHK(ans)                         \
	{                                         \
		gpuAssert((ans), __FILE__, __LINE__); \
	}

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort)
			exit(code);
	}
}

/*
	Calcula el scan inclusivo de cada bloque retorando el resultado parcial de cada uno en offsets_array
*/
__global__ void partial_inclusive_scan(int *d_array, int *out_array, int *offsets_array) {
	int tid            = threadIdx.x;
	int blockOffset    = (blockIdx.x * blockDim.x * 2);
	int blockArraySize = blockDim.x * 2;

	__shared__ int temp[BLOCK_SIZE * 2 * sizeof(int)];

	temp[tid * 2]     = d_array[(tid * 2) + blockOffset];
	temp[2 * tid + 1] = d_array[(tid * 2) + blockOffset + 1];

	__syncthreads();

	int offset = 1;

	for (int i = blockArraySize / 2; i > 0; i = i / 2) {
		if (tid < i) {
			int ai = offset * (2 * tid + 1) - 1;
			int bi = offset * (2 * tid + 2) - 1;
			temp[bi] += temp[ai];
		}
		offset = offset * 2;
		__syncthreads();
	}

	if (tid == 0) {
		temp[blockArraySize - 1] = temp[0];
	}

	for (int i = 1; i < blockArraySize; i = i * 2) {
		offset = offset / 2;

		if (tid < i) {
			int ai = offset * (2 * tid + 1) - 1;
			int bi = offset * (2 * tid + 2) - 1;

			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}

		__syncthreads();
	}

	out_array[(tid * 2) + blockOffset] = temp[2 * tid];
	out_array[(tid * 2) + blockOffset + 1] = temp[2 * tid + 1];

	if (tid == 0) {
		offsets_array[blockIdx.x] = temp[blockArraySize - 1];
	}
}

__global__ void scan_adjust(int *prescan_array, int *scan_array, int *offsets_array, int *initial_value) {
	int tid         = threadIdx.x;
	int blockOffset = (blockIdx.x * blockDim.x * 2);
	int offsetSum   = *initial_value;

	if ((threadIdx.x + blockIdx.x * blockDim.x) == 0) {
		scan_array[0] = *initial_value;
	}

	if (blockIdx.x > 0) {
		offsetSum += offsets_array[blockIdx.x - 1];
	}

	scan_array[tid * 2 + blockOffset + 1]     = prescan_array[tid * 2 + blockOffset] + offsetSum;
	scan_array[2 * tid + 1 + blockOffset + 1] = prescan_array[2 * tid + 1 + blockOffset] + offsetSum;
}

__global__ void exclusive_scan(int *d_array, int *out_array, int *initial_value) {
	size_t input_size    = (ARRAY_SIZE) * sizeof(int);
	int    offsets_count = (ARRAY_SIZE / BLOCK_SIZE / 2);
	size_t size_offsets  = offsets_count * sizeof(int);

	int *prescan_array, *offsets_array, *null_array;
	cudaMalloc((void **)&prescan_array, input_size);
	cudaMalloc((void **)&offsets_array, size_offsets);
	cudaMalloc((void **)&null_array, size_offsets);

	dim3 tamGrid(offsets_count, 1);
	dim3 tamBlock(BLOCK_SIZE, 1);

	partial_inclusive_scan<<<tamGrid, tamBlock>>>(d_array, prescan_array, offsets_array);
	__syncthreads();

	partial_inclusive_scan<<<1, offsets_count>>>(offsets_array, offsets_array, null_array);
	__syncthreads();

	scan_adjust<<<tamGrid, tamBlock>>>(prescan_array, out_array, offsets_array, initial_value);
	__syncthreads();

	cudaFree(prescan_array);
	cudaFree(offsets_array);
	cudaFree(null_array);
}

int main(int argc, char *argv[]) {

	int *h_array, *d_array, *scan_array, *d_initial_value;

	int *h_initial_value = (int *)malloc(sizeof(int));
	*h_initial_value     = INITIAL_VALUE;
	size_t input_size    = (ARRAY_SIZE) * sizeof(int);
	size_t output_size   = (ARRAY_SIZE + 1) * sizeof(int);

	h_array = (int *)malloc(output_size);

	for (int i = 0; i < ARRAY_SIZE; i++)
		h_array[i] = 1;

	CUDA_CHK(cudaMalloc((void **)&d_array, input_size));
	CUDA_CHK(cudaMalloc((void **)&scan_array, output_size));
	CUDA_CHK(cudaMalloc((void **)&d_initial_value, sizeof(int)));

	CUDA_CHK(cudaMemcpy(d_array, h_array, input_size, cudaMemcpyHostToDevice));

	CUDA_CHK(cudaMemcpy(d_initial_value, h_initial_value, sizeof(int), cudaMemcpyHostToDevice));

	for (int i = 0; i < 10; i++) {
		exclusive_scan<<<1, 1>>>(d_array, scan_array, d_initial_value);
		CUDA_CHK(cudaGetLastError());
		CUDA_CHK(cudaDeviceSynchronize());
	}

	CUDA_CHK(cudaMemcpy(h_array, scan_array, output_size, cudaMemcpyDeviceToHost));

	// for (int i = 0; i < ARRAY_SIZE + 1; i++)
	// 	printf("%d \n", h_array[i]);

	free(h_array);
	free(h_initial_value);
	CUDA_CHK(cudaFree(d_array));
	CUDA_CHK(cudaFree(scan_array));
	CUDA_CHK(cudaFree(d_initial_value));

	return 0;
}