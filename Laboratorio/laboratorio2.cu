#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"

#include <nvtx3/nvToolsExt.h>

#define MATRIX_COLUMNS 5000
#define MATRIX_ROWS 4000

#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 32

#define VALUE_TYPE float

__device__ int8_t hA_mas[76][6] =
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

__device__ int8_t hA_menos[76][6] =
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

__device__ int8_t hB_mas[76][7] =
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

__device__ int8_t hB_menos[76][6] =
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

__device__ int8_t hC_mas[20][8] =
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

__device__ int8_t hC_menos[20][8] =
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

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

__device__ void calculo_sub_bloque(VALUE_TYPE *a_sub, VALUE_TYPE *b_sub, VALUE_TYPE *c_partial, VALUE_TYPE *h_results, int id, bool control_if)
{
    // INICIO CALCULO Hs
    int i, idx;

    // control para cuando cantidad de subbloques no es multiplo de cant. bloques x iteracion
    if (id < 76 && control_if)
    {
        VALUE_TYPE parc_a = 0;
        VALUE_TYPE parc_b = 0;

        for (i = 0; i < 6; i++)
        {
            idx = __ldg(&hA_mas[id][i]);
            if (idx == -1)
                break;
            parc_a += a_sub[idx];
        }
        for (i = 0; i < 6; i++)
        {
            idx = __ldg(&hA_menos[id][i]);
            if (idx == -1)
                break;
            parc_a -= a_sub[idx];
        }

        for (i = 0; i < 7; i++)
        {
            idx = __ldg(&hB_mas[id][i]);
            if (idx == -1)
                break;
            parc_b += b_sub[idx];
        }
        for (i = 0; i < 6; i++)
        {
            idx = __ldg(&hB_menos[id][i]);
            if (idx == -1)
                break;
            parc_b -= b_sub[idx];
        }

        h_results[id] = parc_a * parc_b;
    }

    __syncthreads();
    // FIN CALCULO Hs

    if (id < 20 && control_if)
    {
        VALUE_TYPE parc_c = 0;

        for (i = 0; i < 8; i++)
        {
            idx = __ldg(&hC_mas[id][i]);
            if (idx == -1)
                break;
            parc_c += h_results[idx];
        }
        for (i = 0; i < 8; i++)
        {
            idx = __ldg(&hC_menos[id][i]);
            if (idx == -1)
                break;
            parc_c -= h_results[idx];
        }

        atomicAdd(&c_partial[id], parc_c);
    }

    // FIN CALCULO Cs
}

__global__ void calculo_matriz_por_bloque(VALUE_TYPE *a_global, VALUE_TYPE *b_global, VALUE_TYPE *c_global)
{
    __shared__ VALUE_TYPE c_partial[20];
    __shared__ VALUE_TYPE a_stripe[224];  // = (20 + 17) * 6 subbloques paralelos que podemos procesar. + 2 para llevar a multiplo de 32
    __shared__ VALUE_TYPE b_stripe[224];  // = (25 + 12) * 6 subbloques paralelos que podemos procesar
    __shared__ VALUE_TYPE h_results[456]; // = (76 + 20) * 6;

    int sub_bloques = MATRIX_COLUMNS / 5;

    int subblocks_parallel = blockDim.x / 76;

    if (threadIdx.x < 20)
    {
        c_partial[threadIdx.x] = 0;
    }

    // __syncthreads();

    int thread_block = threadIdx.x % (32 * subblocks_parallel);
    int thread_offset = threadIdx.x % 76;

    int salto = thread_block / (subblocks_parallel * 5);

    // calculamos offset en el array global en función de block_row y block_col
    int offset_a = blockIdx.x * (MATRIX_COLUMNS * 4);
    int offset_b = blockIdx.y * (MATRIX_COLUMNS * 5);

    // iteramos en máximo 13 grupos copiando desde memoria global a memoria compartida
    int iterations = (sub_bloques + subblocks_parallel - 1) / subblocks_parallel;

    int i, subblock_in_c_block, subblock_in_loop, shared_ix, offset_iteration_a, offset_iteration_b;
    int ix_a_global, ix_b_global;

    bool a_stripe_bool = threadIdx.x < 20 * subblocks_parallel;                                           // divergen 8 de 192
    bool b_stripe_bool = threadIdx.x >= 32 * subblocks_parallel && threadIdx.x < 57 * subblocks_parallel; // multiplo de 32 para evitar divergencia

    int aux = (thread_block / 5) % subblocks_parallel;

    bool calculo_sub_bloque_bool;
    for (i = 0; i < iterations; i++)
    {
        // subblock_in_c_block = (i * subblocks_parallel) + aux;
        subblock_in_loop = ((i * subblocks_parallel) + aux) % subblocks_parallel;

        // en shared ya dejamos ordenado por bloque para poder referenciar punteros fácilmente
        shared_ix = subblock_in_loop * 37 + (thread_block % 5) + salto * 5;

        offset_iteration_a = i * subblocks_parallel * 5;
        offset_iteration_b = i * subblocks_parallel * 5;

        // cada <cantidad de columnas de subbloques paralelos> le sumamos la diferencia de bloques + el offset de thread
        ix_a_global = offset_a + offset_iteration_a + (thread_block / (subblocks_parallel * 5) * (sub_bloques - subblocks_parallel) * 5) + thread_offset;
        ix_b_global = offset_b + offset_iteration_b + (thread_block / (subblocks_parallel * 5) * (sub_bloques - subblocks_parallel) * 5) + thread_offset;

        if (a_stripe_bool)
        {
            a_stripe[shared_ix] = a_global[ix_a_global];
        }
        else if (b_stripe_bool)
        {
            b_stripe[shared_ix] = b_global[ix_b_global];
        } // divergen 10 de 288 (480 - 192)

        __syncthreads();

        subblock_in_loop = threadIdx.x / 76;

        // calculamos el subbloque de a y b
        VALUE_TYPE *a_sub_bloque = &a_stripe[subblock_in_loop * 37];
        VALUE_TYPE *b_sub_bloque = &b_stripe[subblock_in_loop * 37];
        VALUE_TYPE *h_sub_bloque = &h_results[subblock_in_loop * 76];

        calculo_sub_bloque_bool = (subblock_in_loop + i * subblocks_parallel) < sub_bloques;

        calculo_sub_bloque(a_sub_bloque, b_sub_bloque, c_partial, h_sub_bloque, thread_offset, calculo_sub_bloque_bool);

        __syncthreads(); // esperamos que terminen de calcularse los parciales para el stripe actual antes de iterar nuevamente y sobreescribir la shared memory
    }

    __syncthreads();

    // sumamos los resultados parciales de c_partial en c_global
    if (threadIdx.x < 20)
    {
        int global_idx = blockIdx.y * sub_bloques * 20 + blockIdx.x * 20 + threadIdx.x;
        atomicAdd(&c_global[global_idx], c_partial[threadIdx.x]); // revisar escritura coalesced
    }
}

__global__ void transponer_matriz(VALUE_TYPE *d_matrix, VALUE_TYPE *transposed_matrix)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < MATRIX_COLUMNS && y < MATRIX_COLUMNS)
    {
        int id = y * MATRIX_COLUMNS + x;
        int id_transposed = x * MATRIX_COLUMNS + y;
        transposed_matrix[id_transposed] = d_matrix[id];
    }
}

int main(int argc, char *argv[])
{

    VALUE_TYPE *h_array_a, *h_array_b, *h_array_c, *d_array_a, *d_array_b, *d_array_b_transposed, *d_array_c;

    int size_a = sizeof(VALUE_TYPE) * MATRIX_ROWS * MATRIX_COLUMNS;
    int size_b = sizeof(VALUE_TYPE) * MATRIX_COLUMNS * MATRIX_COLUMNS;

    h_array_a = (VALUE_TYPE *)malloc(size_a);
    h_array_b = (VALUE_TYPE *)malloc(size_b);
    h_array_c = (VALUE_TYPE *)malloc(size_a);

    for (int i = 0; i < MATRIX_ROWS * MATRIX_COLUMNS; i++)
    {
        h_array_a[i] = 1; // + (int)(i / MATRIX_COLUMNS);
        h_array_c[i] = 0;
    }

    for (int i = 0; i < MATRIX_COLUMNS * MATRIX_COLUMNS; i++)
    {
        h_array_b[i] = 1; // + (int)(i / MATRIX_COLUMNS);
    }

    CUDA_CHK(cudaMalloc((void **)&d_array_a, size_a));
    CUDA_CHK(cudaMalloc((void **)&d_array_b, size_b));
    CUDA_CHK(cudaMalloc((void **)&d_array_b_transposed, size_b));
    CUDA_CHK(cudaMalloc((void **)&d_array_c, size_a));

    CUDA_CHK(cudaMemcpy(d_array_a, h_array_a, size_a, cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(d_array_b, h_array_b, size_b, cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(d_array_c, h_array_c, size_a, cudaMemcpyHostToDevice));

    for (int i = 0; i < 1; i++)
    {
        nvtxRangePushA("Calculo de matriz por bloques");
        // transponemos la matriz para optimizar la lectura coalesced
        dim3 tamGridTr((MATRIX_COLUMNS + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (MATRIX_COLUMNS + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);
        dim3 tamBlockTr(BLOCK_SIZE_X, BLOCK_SIZE_Y);
        transponer_matriz<<<tamGridTr, tamBlockTr>>>(d_array_b, d_array_b_transposed);

        CUDA_CHK(cudaGetLastError());
        CUDA_CHK(cudaDeviceSynchronize());

        // int bloques = (MATRIX_COLUMNS / 5) * (MATRIX_ROWS / 4);
        //  lanzamos la grilla con cantidad de bloques en c para aprovechar la shared memory
        //  int tamGrid = bloques;
        dim3 tamGrid(MATRIX_COLUMNS / 5, MATRIX_ROWS / 4);
        // el tamaño de bloque serán 76 hilos mínimo para poder aprovechar la memoria coalesced,
        // y un multiplo de 76 para poder paralelizar lo más posible el producto de cada subbloque
        // cada lectura a memoria son 128 bytes, el tamaño de VALUE_TYPE es 4 bytes,
        // entonces necesitamos multiplos de 32 sin pasarnos de 1024
        // int tamBlock = 76 * (MATRIX_COLUMNS / 5); // nos permite calcular máximo 13 subbloques en paralelo.
        // // TODO Ver tamaño optimo para aprovechar las lecturas coalesced
        int tamBlock = 456;
        /*lectura son 128bytes
        float son 4
        una lectura trae 32 columnas
        32 columnas son 6.4 filas de sub bloques
        76*6 = 456*/
        calculo_matriz_por_bloque<<<tamGrid, tamBlock>>>(d_array_a, d_array_b_transposed, d_array_c);

        CUDA_CHK(cudaGetLastError());
        CUDA_CHK(cudaDeviceSynchronize());
        nvtxRangePop();
    }
    CUDA_CHK(cudaMemcpy(h_array_c, d_array_c, size_a, cudaMemcpyDeviceToHost));

    printf("Primera posicion de h_array_c: %f\n", h_array_c[0]);
    printf("Ultima posicion de h_array_c: %f\n", h_array_c[MATRIX_ROWS * MATRIX_COLUMNS - 1]);

    free(h_array_a);
    free(h_array_b);
    free(h_array_c);

    CUDA_CHK(cudaFree(d_array_a));
    CUDA_CHK(cudaFree(d_array_b));
    CUDA_CHK(cudaFree(d_array_c));

    return 0;
}