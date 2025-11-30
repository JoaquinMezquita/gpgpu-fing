#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "cuda.h"

#define CUDA_CHK(ans)                         \
	{                                         \
		gpuAssert((ans), __FILE__, __LINE__); \
	}
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__device__ int modulo(int a, int b) {
	int r = a % b;
	r = (r < 0) ? r + b : r;
	return r;
}

__global__ void matrix_kernel(int *d_matrix, int n, int length_s, int i1, int j1, int i2, int j2, int val) {
	int id = blockIdx.x * blockDim.x + threadIdx.x; // id del hilo
	if (id < length_s)
	{
		int n_sub = i2 - i1 + 1;					   // largo en x de submatriz
		int col_sub = id / n_sub - 1;				   // columna dentro de la submatriz
		int row_sub = id % n_sub - 1;				   // fila dentro de la submatriz
		int aux = (i1 + row_sub) * n + (j1 + col_sub); // posición en la matriz principal ES:
		// VOY A LA FILA QUE ME TOCA que es i1+row_sub * n y en esa fila voy a la col que me toca que es j1+col_sub
		d_matrix[aux] += val;
	}
}

int main(int argc, char *argv[]) {
	int n, i1, i2, j1, j2, val, size, length_t, length_s;
	int *h_matrix, *d_matrix;

	if (argc < 7) {
		printf("Debe ingresar el tamaño de matriz, i1, j1, i2, j2 y val\n");
	} else {
		n = atoi(argv[1]);
		i1 = atoi(argv[2]);
		j1 = atoi(argv[3]);
		i2 = atoi(argv[4]);
		j2 = atoi(argv[5]);
		val = atoi(argv[6]);

		if (i1 >= i2 || i2 >= n) {
			printf("i1 debe ser menor que i2 y menores que la matriz\n");
			return 0;
		}

		if (j1 >= j2 || j2 >= n) {
			printf("j1 debe ser menor que j2 y menores que la matriz\n");
			return 0;
		}
	}

	length_t = n * n;
	length_s = (i2 - i1 + 1) * (j2 - j1 + 1);

	size = length_t * sizeof(int);
	// reservar memoria para la matriz
	h_matrix = (int *)malloc(size);

	// llenamos la matriz con 0s
	for (int i = 0; i < length_t; i++) {
		h_matrix[i] = 0;
	}

	CUDA_CHK(cudaMalloc((void **)&d_matrix, size));
	CUDA_CHK(cudaMemcpy(d_matrix, h_matrix, size, cudaMemcpyHostToDevice));

	int blockSize = 1024;
	// USAMOS SOLO BLOQUES PARA LA SUBMATRIZ
	int numBlocks = (length_s + blockSize - 1) / blockSize;

	matrix_kernel<<<numBlocks, blockSize>>>(d_matrix, n, length_s, i1, j1, i2, j2, val);

	CUDA_CHK(cudaGetLastError());
	CUDA_CHK(cudaDeviceSynchronize());

	CUDA_CHK(cudaMemcpy(h_matrix, d_matrix, size, cudaMemcpyDeviceToHost));
	CUDA_CHK(cudaFree(d_matrix));

	// Imprimo la matriz
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			printf("%d \t", h_matrix[i * n + j]);
		}
		printf("\n");
	}

	// libero la memoria en la CPU
	free(h_matrix);

	return 0;
}