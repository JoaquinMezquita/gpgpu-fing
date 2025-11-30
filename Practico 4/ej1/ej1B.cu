#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"

#include <cub/cub.cuh>

#include <nvtx3/nvToolsExt.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>

#define ARRAY_SIZE 8192
#define INITIAL_VALUE 0

#define CUDA_CHK(ans)                         \
	{                                         \
		gpuAssert((ans), __FILE__, __LINE__); \
	}

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort)
			exit(code);
	}
}

int main(int argc, char *argv[])
{
	int *h_array;

	size_t input_size = (ARRAY_SIZE) * sizeof(int);

	h_array = (int *)malloc(input_size);

	for (int i = 0; i < ARRAY_SIZE; i++)
		h_array[i] = 1;

	thrust::device_vector<int> d_input(h_array, h_array + ARRAY_SIZE);
	thrust::device_vector<int> d_output(ARRAY_SIZE);

	for (int i = 0; i < 10; i++)
	{
		nvtxRangePushA("thrust exclusive_scan");
		thrust::exclusive_scan(d_input.begin(), d_input.end(), d_output.begin(), INITIAL_VALUE);
		CUDA_CHK(cudaGetLastError());
		CUDA_CHK(cudaDeviceSynchronize());
		nvtxRangePop();
	}

	thrust::host_vector<int> h_output = d_output;

	void *d_temp_storage = nullptr;
	size_t temp_storage_bytes = 0;

	int *d_array, *scan_array, *d_initial_value;
	CUDA_CHK(cudaMalloc((void **)&d_array, input_size));
	CUDA_CHK(cudaMalloc((void **)&scan_array, input_size));

	CUDA_CHK(cudaMemcpy(d_array, h_array, input_size, cudaMemcpyHostToDevice));

	cub::DeviceScan::ExclusiveScan(
		d_temp_storage, temp_storage_bytes,
		d_array, scan_array, cub::Sum(), INITIAL_VALUE, ARRAY_SIZE);

	CUDA_CHK(cudaMalloc(&d_temp_storage, temp_storage_bytes));

	for (int i = 0; i < 10; i++)
	{
		nvtxRangePushA("cub exclusive_scan");
		cub::DeviceScan::ExclusiveScan(
			d_temp_storage, temp_storage_bytes,
			d_array, scan_array, cub::Sum(), INITIAL_VALUE, ARRAY_SIZE);

		CUDA_CHK(cudaGetLastError());
		CUDA_CHK(cudaDeviceSynchronize());
		nvtxRangePop();
	}

	CUDA_CHK(cudaMemcpy(h_array, scan_array, input_size, cudaMemcpyDeviceToHost));
	// for (int i = 0; i < ARRAY_SIZE; i++)
	// {
	// 	printf("%d\n", h_array[i]);
	// }

	free(h_array);
	CUDA_CHK(cudaFree(d_array));
	CUDA_CHK(cudaFree(scan_array));
	CUDA_CHK(cudaFree(d_temp_storage));

	return 0;
}