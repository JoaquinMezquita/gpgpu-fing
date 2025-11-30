#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"

#include <nvtx3/nvToolsExt.h>
#include <cublas_v2.h>

#define MATRIX_COLUMNS 1000
#define MATRIX_ROWS 800

#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 32

#define VALUE_TYPE float

__device__ __constant__ int8_t hA_mas[76][6] =
    {{11, -1, -1, -1, -1, -1},
     {6, 9, -1, -1, -1, -1},
     {16, -1, -1, -1, -1, -1},
     {1, 3, 13, -1, -1, -1},
     {4, 6, 9, -1, -1, -1},
     {-1, -1, -1, -1, -1, -1},
     {15, -1, -1, -1, -1, -1},
     {11, -1, -1, -1, -1, -1},
     {18, -1, -1, -1, -1, -1},
     {6, 9, -1, -1, -1, -1},
     {16, -1, -1, -1, -1, -1},
     {15, -1, -1, -1, -1, -1},
     {1, 3, 8, -1, -1, -1},
     {2, 12, -1, -1, -1, -1},
     {-1, -1, -1, -1, -1, -1},
     {12, -1, -1, -1, -1, -1},
     {1, 3, 6, 8, 12, 16},
     {5, -1, -1, -1, -1, -1},
     {-1, -1, -1, -1, -1, -1},
     {5, 7, -1, -1, -1, -1},
     {5, 7, -1, -1, -1, -1},
     {2, -1, -1, -1, -1, -1},
     {2, -1, -1, -1, -1, -1},
     {4, -1, -1, -1, -1, -1},
     {-1, -1, -1, -1, -1, -1},
     {3, 4, -1, -1, -1, -1},
     {2, 12, -1, -1, -1, -1},
     {-1, -1, -1, -1, -1, -1},
     {10, -1, -1, -1, -1, -1},
     {10, 13, -1, -1, -1, -1},
     {-1, -1, -1, -1, -1, -1},
     {5, 15, 18, -1, -1, -1},
     {17, -1, -1, -1, -1, -1},
     {18, -1, -1, -1, -1, -1},
     {-1, -1, -1, -1, -1, -1},
     {7, -1, -1, -1, -1, -1},
     {19, -1, -1, -1, -1, -1},
     {12, -1, -1, -1, -1, -1},
     {19, -1, -1, -1, -1, -1},
     {3, 4, -1, -1, -1, -1},
     {15, -1, -1, -1, -1, -1},
     {9, -1, -1, -1, -1, -1},
     {8, -1, -1, -1, -1, -1},
     {7, 11, -1, -1, -1, -1},
     {13, -1, -1, -1, -1, -1},
     {-1, -1, -1, -1, -1, -1},
     {5, 14, -1, -1, -1, -1},
     {12, -1, -1, -1, -1, -1},
     {3, 4, 8, 9, -1, -1},
     {-1, -1, -1, -1, -1, -1},
     {6, -1, -1, -1, -1, -1},
     {16, -1, -1, -1, -1, -1},
     {-1, -1, -1, -1, -1, -1},
     {1, 3, 12, 17, -1, -1},
     {3, -1, -1, -1, -1, -1},
     {0, 19, -1, -1, -1, -1},
     {-1, -1, -1, -1, -1, -1},
     {-1, -1, -1, -1, -1, -1},
     {13, 18, -1, -1, -1, -1},
     {9, 19, -1, -1, -1, -1},
     {3, 13, -1, -1, -1, -1},
     {5, 15, -1, -1, -1, -1},
     {-1, -1, -1, -1, -1, -1},
     {0, 10, -1, -1, -1, -1},
     {15, -1, -1, -1, -1, -1},
     {0, 2, 12, 16, -1, -1},
     {9, -1, -1, -1, -1, -1},
     {0, 2, 18, 19, -1, -1},
     {3, 8, -1, -1, -1, -1},
     {7, 17, -1, -1, -1, -1},
     {12, 14, 17, 19, -1, -1},
     {-1, -1, -1, -1, -1, -1},
     {2, 7, -1, -1, -1, -1},
     {5, 8, 12, -1, -1, -1},
     {6, 9, 10, 15, -1, -1},
     {2, 12, -1, -1, -1, -1}};

__device__ __constant__ int8_t hA_menos[76][6] =
    {{-1, -1, -1, -1, -1, -1},
     {14, -1, -1, -1, -1, -1},
     {10, 15, -1, -1, -1, -1},
     {-1, -1, -1, -1, -1, -1},
     {-1, -1, -1, -1, -1, -1},
     {6, 9, 19, -1, -1, -1},
     {0, 16, -1, -1, -1, -1},
     {12, 17, -1, -1, -1, -1},
     {1, 3, -1, -1, -1, -1},
     {-1, -1, -1, -1, -1, -1},
     {5, 15, -1, -1, -1, -1},
     {16, -1, -1, -1, -1, -1},
     {-1, -1, -1, -1, -1, -1},
     {11, -1, -1, -1, -1, -1},
     {1, 3, -1, -1, -1, -1},
     {11, -1, -1, -1, -1, -1},
     {5, 7, 11, 15, -1, -1},
     {-1, -1, -1, -1, -1, -1},
     {7, -1, -1, -1, -1, -1},
     {4, 9, -1, -1, -1, -1},
     {9, -1, -1, -1, -1, -1},
     {3, 8, -1, -1, -1, -1},
     {-1, -1, -1, -1, -1, -1},
     {-1, -1, -1, -1, -1, -1},
     {0, -1, -1, -1, -1, -1},
     {2, -1, -1, -1, -1, -1},
     {10, -1, -1, -1, -1, -1},
     {13, -1, -1, -1, -1, -1},
     {-1, -1, -1, -1, -1, -1},
     {12, -1, -1, -1, -1, -1},
     {3, 4, 13, -1, -1, -1},
     {-1, -1, -1, -1, -1, -1},
     {-1, -1, -1, -1, -1, -1},
     {-1, -1, -1, -1, -1, -1},
     {19, -1, -1, -1, -1, -1},
     {9, 19, -1, -1, -1, -1},
     {15, 18, -1, -1, -1, -1},
     {7, 10, 13, -1, -1, -1},
     {10, 15, 18, -1, -1, -1},
     {2, 18, -1, -1, -1, -1},
     {0, 19, -1, -1, -1, -1},
     {5, 14, -1, -1, -1, -1},
     {-1, -1, -1, -1, -1, -1},
     {12, -1, -1, -1, -1, -1},
     {12, 17, -1, -1, -1, -1},
     {14, -1, -1, -1, -1, -1},
     {9, 10, -1, -1, -1, -1},
     {7, -1, -1, -1, -1, -1},
     {0, 2, 5, 7, -1, -1},
     {3, 8, -1, -1, -1, -1},
     {-1, -1, -1, -1, -1, -1},
     {-1, -1, -1, -1, -1, -1},
     {1, -1, -1, -1, -1, -1},
     {6, 9, 11, 16, 18, 19},
     {18, -1, -1, -1, -1, -1},
     {4, 15, -1, -1, -1, -1},
     {10, 15, -1, -1, -1, -1},
     {3, 4, 13, 14, -1, -1},
     {12, 17, -1, -1, -1, -1},
     {-1, -1, -1, -1, -1, -1},
     {-1, -1, -1, -1, -1, -1},
     {-1, -1, -1, -1, -1, -1},
     {12, 17, -1, -1, -1, -1},
     {2, 3, 12, 13, -1, -1},
     {0, -1, -1, -1, -1, -1},
     {1, 4, 6, 9, 11, 15},
     {14, -1, -1, -1, -1, -1},
     {3, 4, 15, 17, -1, -1},
     {2, 7, -1, -1, -1, -1},
     {9, 19, -1, -1, -1, -1},
     {10, 13, 15, 18, -1, -1},
     {5, 8, 15, 18, -1, -1},
     {3, 4, 8, 9, -1, -1},
     {7, 10, 13, -1, -1, -1},
     {1, 3, 11, 13, 14, 16},
     {-1, -1, -1, -1, -1, -1}};

__device__ __constant__ int8_t hB_mas[76][7] =
    {{-1, -1, -1, -1, -1, -1, -1},
     {-1, -1, -1, -1, -1, -1, -1},
     {21, -1, -1, -1, -1, -1, -1},
     {-1, -1, -1, -1, -1, -1, -1},
     {4, -1, -1, -1, -1, -1, -1},
     {11, 4, -1, -1, -1, -1, -1},
     {0, 16, -1, -1, -1, -1, -1},
     {2, -1, -1, -1, -1, -1, -1},
     {11, 3, -1, -1, -1, -1, -1},
     {4, -1, -1, -1, -1, -1, -1},
     {6, -1, -1, -1, -1, -1, -1},
     {0, -1, -1, -1, -1, -1, -1},
     {6, 3, -1, -1, -1, -1, -1},
     {16, 2, -1, -1, -1, -1, -1},
     {3, -1, -1, -1, -1, -1, -1},
     {2, -1, -1, -1, -1, -1, -1},
     {6, -1, -1, -1, -1, -1, -1},
     {0, 5, 9, -1, -1, -1, -1},
     {2, 7, 9, -1, -1, -1, -1},
     {15, -1, -1, -1, -1, -1, -1},
     {9, -1, -1, -1, -1, -1, -1},
     {0, 5, 17, 18, -1, -1, -1},
     {17, 18, -1, -1, -1, -1, -1},
     {19, -1, -1, -1, -1, -1, -1},
     {0, -1, -1, -1, -1, -1, -1},
     {18, -1, -1, -1, -1, -1, -1},
     {0, 20, 22, -1, -1, -1, -1},
     {-1, -1, -1, -1, -1, -1, -1},
     {0, 20, 22, -1, -1, -1, -1},
     {22, -1, -1, -1, -1, -1, -1},
     {19, -1, -1, -1, -1, -1, -1},
     {10, -1, -1, -1, -1, -1, -1},
     {-1, -1, -1, -1, -1, -1, -1},
     {3, 13, -1, -1, -1, -1, -1},
     {10, 4, 14, -1, -1, -1, -1},
     {2, 7, 12, 9, -1, -1, -1},
     {10, -1, -1, -1, -1, -1, -1},
     {22, 3, 8, 23, -1, -1, -1},
     {10, 4, 14, 24, -1, -1, -1},
     {17, 18, -1, -1, -1, -1, -1},
     {10, 2, 12, 4, 14, -1, -1},
     {3, 8, 23, -1, -1, -1, -1},
     {3, 8, -1, -1, -1, -1, -1},
     {6, -1, -1, -1, -1, -1, -1},
     {22, 3, 13, 23, 4, 14, 24},
     {-1, -1, -1, -1, -1, -1, -1},
     {0, 5, 20, -1, -1, -1, -1},
     {6, 7, 22, 3, 8, 23, -1},
     {15, -1, -1, -1, -1, -1, -1},
     {6, 17, 18, -1, -1, -1, -1},
     {1, 6, -1, -1, -1, -1, -1},
     {0, 1, 11, -1, -1, -1, -1},
     {16, 3, -1, -1, -1, -1, -1},
     {11, -1, -1, -1, -1, -1, -1},
     {2, 12, 13, -1, -1, -1, -1},
     {2, 12, 4, 14, -1, -1, -1},
     {-1, -1, -1, -1, -1, -1, -1},
     {19, -1, -1, -1, -1, -1, -1},
     {3, 13, 23, 4, 14, 24, -1},
     {11, -1, -1, -1, -1, -1, -1},
     {0, 20, 23, 19, -1, -1, -1},
     {5, 10, 6, -1, -1, -1, -1},
     {-1, -1, -1, -1, -1, -1, -1},
     {0, 20, -1, -1, -1, -1, -1},
     {15, 16, 19, -1, -1, -1, -1},
     {16, -1, -1, -1, -1, -1, -1},
     {0, 5, 20, 9, 24, -1, -1},
     {17, -1, -1, -1, -1, -1, -1},
     {17, 19, -1, -1, -1, -1, -1},
     {-1, -1, -1, -1, -1, -1, -1},
     {-1, -1, -1, -1, -1, -1, -1},
     {3, 8, 13, -1, -1, -1, -1},
     {0, 5, 16, 9, -1, -1, -1},
     {3, 8, 23, -1, -1, -1, -1},
     {21, -1, -1, -1, -1, -1, -1},
     {15, 16, 17, -1, -1, -1, -1}};

__device__ __constant__ int8_t hB_menos[76][6] =
    {{1, 21, 2, -1, -1, -1},
     {21, 4, -1, -1, -1, -1},
     {0, -1, -1, -1, -1, -1},
     {21, 3, -1, -1, -1, -1},
     {16, -1, -1, -1, -1, -1},
     {-1, -1, -1, -1, -1, -1},
     {-1, -1, -1, -1, -1, -1},
     {11, -1, -1, -1, -1, -1},
     {-1, -1, -1, -1, -1, -1},
     {-1, -1, -1, -1, -1, -1},
     {0, -1, -1, -1, -1, -1},
     {-1, -1, -1, -1, -1, -1},
     {-1, -1, -1, -1, -1, -1},
     {-1, -1, -1, -1, -1, -1},
     {-1, -1, -1, -1, -1, -1},
     {-1, -1, -1, -1, -1, -1},
     {-1, -1, -1, -1, -1, -1},
     {-1, -1, -1, -1, -1, -1},
     {-1, -1, -1, -1, -1, -1},
     {0, 5, 9, -1, -1, -1},
     {-1, -1, -1, -1, -1, -1},
     {15, 2, 7, -1, -1, -1},
     {2, -1, -1, -1, -1, -1},
     {18, 4, -1, -1, -1, -1},
     {15, -1, -1, -1, -1, -1},
     {-1, -1, -1, -1, -1, -1},
     {15, -1, -1, -1, -1, -1},
     {22, 3, 23, -1, -1, -1},
     {-1, -1, -1, -1, -1, -1},
     {-1, -1, -1, -1, -1, -1},
     {18, 4, 24, -1, -1, -1},
     {3, 8, 13, -1, -1, -1},
     {2, 12, -1, -1, -1, -1},
     {10, -1, -1, -1, -1, -1},
     {-1, -1, -1, -1, -1, -1},
     {-1, -1, -1, -1, -1, -1},
     {-1, -1, -1, -1, -1, -1},
     {-1, -1, -1, -1, -1, -1},
     {-1, -1, -1, -1, -1, -1},
     {2, 12, -1, -1, -1, -1},
     {17, 19, -1, -1, -1, -1},
     {0, 5, 20, 9, -1, -1},
     {-1, -1, -1, -1, -1, -1},
     {2, -1, -1, -1, -1, -1},
     {-1, -1, -1, -1, -1, -1},
     {4, 24, -1, -1, -1, -1},
     {3, 8, 23, -1, -1, -1},
     {-1, -1, -1, -1, -1, -1},
     {0, 5, -1, -1, -1, -1},
     {2, 7, 8, -1, -1, -1},
     {4, -1, -1, -1, -1, -1},
     {-1, -1, -1, -1, -1, -1},
     {1, -1, -1, -1, -1, -1},
     {-1, -1, -1, -1, -1, -1},
     {11, 17, 18, -1, -1, -1},
     {17, 19, -1, -1, -1, -1},
     {10, 20, 21, 4, 14, 24},
     {4, 24, -1, -1, -1, -1},
     {-1, -1, -1, -1, -1, -1},
     {2, 7, 12, 9, 14, -1},
     {15, 21, 18, 4, 24, -1},
     {3, 8, 13, -1, -1, -1},
     {11, 12, 22, 3, 13, 23},
     {15, -1, -1, -1, -1, -1},
     {10, 4, 14, -1, -1, -1},
     {-1, -1, -1, -1, -1, -1},
     {21, 3, 8, 23, -1, -1},
     {2, 12, -1, -1, -1, -1},
     {16, 2, 7, 9, -1, -1},
     {2, 7, 12, -1, -1, -1},
     {4, 14, 24, -1, -1, -1},
     {-1, -1, -1, -1, -1, -1},
     {15, 19, -1, -1, -1, -1},
     {-1, -1, -1, -1, -1, -1},
     {-1, -1, -1, -1, -1, -1},
     {0, 20, 22, -1, -1, -1}};

__device__ __constant__ int8_t hC_mas[20][8] =
    {{11, 13, 52, 4, -1, -1, -1, -1},
     {12, 14, 19, 20, 22, 24, 48, 49},
     {14, 22, 23, 33, 39, 54, -1, -1},
     {11, 13, 22, 23, 24, 25, 4, -1},
     {14, 23, 24, 26, 29, 30, 60, 63},
     {9, 10, 12, 14, 15, 50, -1, -1},
     {11, 16, 17, 42, 43, -1, -1, -1},
     {18, 31, 34, 35, 36, -1, -1, -1},
     {9, 17, 19, 72, -1, -1, -1, -1},
     {41, 45, 66, 73, -1, -1, -1, -1},
     {9, 14, 15, 1, 2, 74, -1, -1},
     {41, 43, 47, -1, -1, -1, -1, -1},
     {32, 36, 44, 62, -1, -1, -1, -1},
     {15, 26, 28, 30, 45, 75, -1, -1},
     {11, 27, 28, 45, 3, -1, -1, -1},
     {11, 51, 53, 8, -1, -1, -1, -1},
     {10, 20, 32, 61, -1, -1, -1, -1},
     {9, 14, 15, 33, 5, 7, -1, -1},
     {11, 24, 25, 40, 64, -1, -1, -1},
     {29, 34, 38, 2, 56, 58, -1, -1}};

__device__ __constant__ int8_t hC_menos[20][8] =
    {{9, 14, 15, 65, 6, -1, -1, -1},
     {21, 42, -1, -1, -1, -1, -1, -1},
     {36, 40, 55, 8, -1, -1, -1, -1},
     {9, 15, 65, 6, -1, -1, -1, -1},
     {27, 3, -1, -1, -1, -1, -1, -1},
     {11, 16, 43, -1, -1, -1, -1, -1},
     {10, 12, 14, 15, 18, 20, -1, -1},
     {9, 42, 59, 5, 71, -1, -1, -1},
     {18, 21, 23, 25, 4, 68, -1, -1},
     {9, 17, 1, 29, 37, 42, -1, -1},
     {11, 0, 3, -1, -1, -1, -1, -1},
     {15, 18, 20, 27, 28, 37, 46, -1},
     {15, 27, 38, 45, 70, 7, -1, -1},
     {13, 22, 25, 57, -1, -1, -1, -1},
     {9, 14, 1, 29, 2, 74, -1, -1},
     {9, 14, 15, 5, 7, -1, -1, -1},
     {11, 17, 31, 33, 35, 69, -1, -1},
     {32, 34, 36, 53, 8, -1, -1, -1},
     {32, 34, 39, 67, 6, -1, -1, -1},
     {11, 28, 33, 44, -1, -1, -1, -1}};

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

__device__ VALUE_TYPE *obtener_bloque(VALUE_TYPE *base, int bloque_idx, int subbloques_por_bloque, int size) {
    return &base[bloque_idx * subbloques_por_bloque * size];
}

__device__ VALUE_TYPE *obtener_subbloque(VALUE_TYPE *bloque_ptr, int subbloque_idx, int size) {
    return &bloque_ptr[subbloque_idx * size];
}

__device__ void calculo_sub_bloque(VALUE_TYPE *a_sub, VALUE_TYPE *b_sub, VALUE_TYPE *c_partial, int sub_bloque, int id, int iteration) {
    // INICIO CALCULO Hs

    __shared__ VALUE_TYPE h_results[988]; // 988 = 76 * 13;

    int hOffset = (sub_bloque * 76);

    if ((sub_bloque + iteration * blockDim.x / 76) < (MATRIX_COLUMNS / 5)) {
        if (id < 76) {
            VALUE_TYPE parc_a = 0;
            VALUE_TYPE parc_b = 0;

            for (int i = 0; i < 6; i++) {
                int idx = hA_mas[id][i];
                if (idx == -1)
                    break;
                parc_a += a_sub[idx];
            }
            for (int i = 0; i < 6; i++) {
                int idx = hA_menos[id][i];
                if (idx == -1)
                    break;
                parc_a -= a_sub[idx];
            }

            for (int i = 0; i < 7; i++) {
                int idx = hB_mas[id][i];
                if (idx == -1)
                    break;
                parc_b += b_sub[idx];
            }
            for (int i = 0; i < 6; i++) {
                int idx = hB_menos[id][i];
                if (idx == -1)
                    break;
                parc_b -= b_sub[idx];
            }
            h_results[hOffset + id] = parc_a * parc_b;
        }
    }

    __syncthreads();
    // FIN CALCULO Hs

    if ((sub_bloque + iteration * blockDim.x / 76) < (MATRIX_COLUMNS / 5)) {

        if (id < (MATRIX_ROWS * MATRIX_COLUMNS)) {
            VALUE_TYPE parc_c = 0;

            for (int i = 0; i < 8; i++) {
                int idx = hC_mas[id][i];
                if (idx == -1)
                    break;
                parc_c += h_results[idx];
            }
            for (int i = 0; i < 8; i++) {
                int idx = hC_menos[id][i];
                if (idx == -1)
                    break;
                parc_c -= h_results[idx];
            }

            atomicAdd(&c_partial[id], parc_c);
        }
    }

    // FIN CALCULO Cs
}

__global__ void calculo_matriz_por_bloque(VALUE_TYPE *a_global, VALUE_TYPE *b_global, VALUE_TYPE *c_global) {
    int sub_bloques        = MATRIX_COLUMNS / 5;
    int block_row          = blockIdx.x / sub_bloques;
    int block_col          = blockIdx.x % sub_bloques;
    int subblocks_parallel = blockDim.x / 76;

    __shared__ VALUE_TYPE c_partial[20];
    __shared__ VALUE_TYPE a_stripe[260]; // = 20 * 13 subbloques paralelos que podemos procesar. TODO asignar dinámicamente
    __shared__ VALUE_TYPE b_stripe[325]; // = 25 * 13 subbloques paralelos que podemos procesar

    if (threadIdx.x < 20)
        c_partial[threadIdx.x] = 0;
    
    __syncthreads();

    // iteramos en máximo 13 grupos copiando desde memoria global a memoria compartida
    int iterations = (sub_bloques + subblocks_parallel - 1) / subblocks_parallel;
    for (int i = 0; i < iterations; i++)  {
        int subblock_in_c_block = (i * subblocks_parallel) + (threadIdx.x / 5) % subblocks_parallel;
        int subblock_in_loop = subblock_in_c_block % subblocks_parallel;

        int thread_offset = threadIdx.x % 76;

        // Ajustar thread_offset para que coincida con subblock_in_c_block
        // Cuando subblock_in_c_block es 0 o 1, thread_offset debe ser igual a subblock_in_c_block * 76 + (threadIdx.x % 76)

        if (threadIdx.x < 25 * subblocks_parallel) {
            // en shared ya dejamos ordenado por bloque para poder referenciar punteros fácilmente
            int salto = threadIdx.x / (subblocks_parallel * 5);

            int shared_a_ix = subblock_in_loop * 20 + (threadIdx.x % 5) + salto * 5;
            int shared_b_ix = subblock_in_loop * 25 + (threadIdx.x % 5) + salto * 5;

            // calculamos offset en el array global en función de block_row y block_col
            int offset_a = block_row * (MATRIX_COLUMNS * 4);
            int offset_b = block_col * (MATRIX_COLUMNS * 5);

            int offset_iteration_a = i * subblocks_parallel * 5;
            int offset_iteration_b = i * subblocks_parallel * 5;

            // cada <cantidad de columnas de subbloques paralelos> le sumamos la diferencia de bloques + el offset de thread
            int ix_a_global = offset_a + offset_iteration_a + (threadIdx.x / (subblocks_parallel * 5) * (sub_bloques - subblocks_parallel) * 5) + thread_offset;
            int ix_b_global = offset_b + offset_iteration_b + (threadIdx.x / (subblocks_parallel * 5) * (sub_bloques - subblocks_parallel) * 5) + thread_offset;

            if (threadIdx.x < 20 * subblocks_parallel)
                a_stripe[shared_a_ix] = a_global[ix_a_global];
            

            if (threadIdx.x < 25 * subblocks_parallel)
                b_stripe[shared_b_ix] = b_global[ix_b_global];
            
        }
        __syncthreads();

        subblock_in_loop = threadIdx.x / 76;

        // calculamos el subbloque de a y b
        VALUE_TYPE *a_sub_bloque = obtener_subbloque(a_stripe, subblock_in_loop, 20);
        VALUE_TYPE *b_sub_bloque = obtener_subbloque(b_stripe, subblock_in_loop, 25);

        calculo_sub_bloque(a_sub_bloque, b_sub_bloque, c_partial, subblock_in_loop, thread_offset, i);

        __syncthreads(); // esperamos que terminen de calcularse los parciales para el stripe actual antes de iterar nuevamente y sobreescribir la shared memory
    }

    __syncthreads();

    // sumamos los resultados parciales de c_partial en c_global
    if (threadIdx.x < 20) {
        int global_idx = block_row * (MATRIX_COLUMNS / 5) * 20 + block_col * 20 + threadIdx.x;
        atomicAdd(&c_global[global_idx], c_partial[threadIdx.x]); // revisar escritura coalesced
    }
}

__global__ void transponer_matriz(VALUE_TYPE *d_matrix, VALUE_TYPE *transposed_matrix) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < MATRIX_COLUMNS && y < MATRIX_COLUMNS) {
        int id            = y * MATRIX_COLUMNS + x;
        int id_transposed = x * MATRIX_COLUMNS + y;

        transposed_matrix[id_transposed] = d_matrix[id];
    }
}

int main(int argc, char *argv[]) {

    VALUE_TYPE *h_array_a, *h_array_b, *h_array_c, *d_array_a, *d_array_b, *d_array_c;

    int size_a = sizeof(VALUE_TYPE) * MATRIX_ROWS    * MATRIX_COLUMNS;
    int size_b = sizeof(VALUE_TYPE) * MATRIX_COLUMNS * MATRIX_COLUMNS;

    h_array_a  = (VALUE_TYPE *)malloc(size_a);
    h_array_b  = (VALUE_TYPE *)malloc(size_b);
    h_array_c  = (VALUE_TYPE *)malloc(size_a);

    for (int i = 0; i < MATRIX_ROWS * MATRIX_COLUMNS; i++) {
        if (i % 2 == 0) {
             h_array_a[i] = 1; // + (int)(i / MATRIX_COLUMNS);
        } else {
             h_array_a[i] = 0; // + (int)(i / MATRIX_COLUMNS);
             h_array_c[i] = 0;
        }
    }

    for (int i = 0; i < MATRIX_COLUMNS * MATRIX_COLUMNS; i++) {
        if (i % 2 == 0)
            h_array_b[i] = 1; // + (int)(i / MATRIX_COLUMNS);
        else 
            h_array_b[i] = 0; // + (int)(i / MATRIX_COLUMNS);
    }

    CUDA_CHK(cudaMalloc((void **)&d_array_a, size_a));
    CUDA_CHK(cudaMalloc((void **)&d_array_b, size_b));
    CUDA_CHK(cudaMalloc((void **)&d_array_c, size_a));

    CUDA_CHK(cudaMemcpy(d_array_a, h_array_a, size_a, cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(d_array_b, h_array_b, size_b, cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(d_array_c, h_array_c, size_a, cudaMemcpyHostToDevice));

    // Reserva memoria adicional para el kernel nuestro
    VALUE_TYPE *d_array_b_transposed, *d_array_custom;
    CUDA_CHK(cudaMalloc((void**)&d_array_b_transposed, size_b));
    CUDA_CHK(cudaMalloc((void**)&d_array_custom, size_a));

    // Lanza kernels personalizados (transposición + cálculo)
    dim3 tamGridTr((MATRIX_COLUMNS + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X,
                   (MATRIX_COLUMNS + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);
    dim3 tamBlockTr(BLOCK_SIZE_X, BLOCK_SIZE_Y);

    transponer_matriz<<<tamGridTr, tamBlockTr>>>(d_array_b, 
                                                 d_array_b_transposed);
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaDeviceSynchronize());

    int bloques  = (MATRIX_COLUMNS / 5) * (MATRIX_ROWS / 4);
    int tamBlock = 76 * (MATRIX_COLUMNS / 5);

    if (tamBlock > 1024)
        tamBlock = 988;

    calculo_matriz_por_bloque<<<bloques, tamBlock>>>(d_array_a,
                                                     d_array_b_transposed,
                                                     d_array_custom);
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaDeviceSynchronize());

    // Copia resultado personalizado a CPU e imprime
    CUDA_CHK(cudaMemcpy(h_array_c, d_array_custom, size_a, cudaMemcpyDeviceToHost));

    //print de toda la matriz
    for (int i = 0; i < MATRIX_ROWS * MATRIX_COLUMNS; i++) 
        printf("h_array_c[%d]: %f\n", i, h_array_c[i]);

    CUDA_CHK(cudaMemcpy(d_array_a, h_array_a, size_a, cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(d_array_b, h_array_b, size_b, cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(d_array_c, h_array_c, size_a, cudaMemcpyHostToDevice));

    const VALUE_TYPE alpha = 1.0f;
    const VALUE_TYPE beta = 0.0f;
    cublasHandle_t handle;
    // Create cuBLAS handle
    cublasCreate(&handle);
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHK(cudaEventCreate(&start));
    CUDA_CHK(cudaEventCreate(&stop));
    // Record start event
    CUDA_CHK(cudaEventRecord(start, 0));
    // Perform matrix multiplication with cuBLAS
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T,
                MATRIX_ROWS, MATRIX_COLUMNS, MATRIX_COLUMNS,
                &alpha,
                d_array_b, MATRIX_COLUMNS,
                d_array_a, MATRIX_ROWS,
                &beta,
                d_array_c, MATRIX_COLUMNS);
    // Record stop event and synchronize
    CUDA_CHK(cudaEventRecord(stop, 0));
    CUDA_CHK(cudaEventSynchronize(stop));
    // Calculate elapsed time
    float elapsed_ms = 0.0f;
    CUDA_CHK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    // Print timing result
    //print de toda la matriz

    // Clean up events
    CUDA_CHK(cudaEventDestroy(start));
    CUDA_CHK(cudaEventDestroy(stop));
    // Destroy cuBLAS handle
    cublasDestroy(handle);

    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaDeviceSynchronize());

    CUDA_CHK(cudaMemcpy(h_array_c, d_array_c, size_a, cudaMemcpyDeviceToHost));

    // CALCULO CON CUBLAS
    printf("CALCULO CON CUBLAS\n");
    for (int i = 0; i < MATRIX_ROWS * MATRIX_COLUMNS; i++)
        printf("h_array_c[%d]: %f\n", i, h_array_c[i]);

    // Calcula norma de la diferencia: d_array_c (cuBLAS) - d_array_custom
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    const VALUE_TYPE neg_one = -1.0f;
    cublasSaxpy(cublas_handle,
                MATRIX_ROWS * MATRIX_COLUMNS,
                &neg_one,
                d_array_custom, 1,
                d_array_c, 1);

    float diff_norm = 0.0f;
    cublasSnrm2(cublas_handle,
                MATRIX_ROWS * MATRIX_COLUMNS,
                d_array_c, 1,
                &diff_norm);

    printf("Norma de la diferencia: %f\n", diff_norm);
    cublasDestroy(cublas_handle);

    // Libera buffers personalizados
    CUDA_CHK(cudaFree(d_array_b_transposed));
    CUDA_CHK(cudaFree(d_array_custom));

    free(h_array_a);
    free(h_array_b);
    free(h_array_c);

    CUDA_CHK(cudaFree(d_array_a));
    CUDA_CHK(cudaFree(d_array_b));
    CUDA_CHK(cudaFree(d_array_c));

    return 0;
}