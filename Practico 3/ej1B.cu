#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"

#define ROWS 4096
#define COLUMNS 4096
#define BLOCK_SIZE_X 2
#define BLOCK_SIZE_Y 32

#define CUDA_CHK(ans)                         \
	{                                         \
		gpuAssert((ans), __FILE__, __LINE__); \
	}

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort)
			exit(code);
	}
}

__global__ void transpose_kernel(int *d_matrix, int *transposed_matrix) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int id            = COLUMNS * y + x;
	int id_transposed = ROWS    * x + y;

	transposed_matrix[id_transposed] = d_matrix[id];
}

int main(int argc, char *argv[]) {

	int *h_matrix, *d_matrix, *transposed_matrix;

	size_t size = ROWS * COLUMNS * sizeof(int);

	h_matrix = (int *)malloc(size);

	for (int i = 0; i < ROWS * COLUMNS; i++)
		h_matrix[i] = i;

	CUDA_CHK(cudaMalloc((void **)&d_matrix, size));
	CUDA_CHK(cudaMalloc((void **)&transposed_matrix, size));

	CUDA_CHK(cudaMemcpy(d_matrix, h_matrix, size, cudaMemcpyHostToDevice));

	dim3 tamGrid(ROWS / BLOCK_SIZE_X, COLUMNS / BLOCK_SIZE_Y);
	dim3 tamBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);

	for (int i = 0; i < 10; i++) {
		transpose_kernel<<<tamGrid, tamBlock>>>(d_matrix, transposed_matrix);
		CUDA_CHK(cudaGetLastError());
		CUDA_CHK(cudaDeviceSynchronize());
	}

	CUDA_CHK(cudaMemcpy(h_matrix, transposed_matrix, size, cudaMemcpyDeviceToHost));

	free(h_matrix);
	CUDA_CHK(cudaFree(d_matrix));
	CUDA_CHK(cudaFree(transposed_matrix));

	return 0;
}