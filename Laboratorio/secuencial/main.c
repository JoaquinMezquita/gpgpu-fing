#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define M 400   // Must be multiple of 4
#define K 500  // Must be multiple of 5
#define N 500  // Must be multiple of 5

void deepmatmul(float a[4][5], float b[5][5], float c[4][5]);
//void deepmatmul2(float a[4][5], float b[5][5], float c[4][5]);
void deepmatmul3(float a[20], float b[25], float c[20], 
                 int hA_mas[76][6], int hA_menos[76][6], 
                 int hB_mas[76][7], int hB_menos[76][6], 
                 int hC_mas[20][8], int hC_menos[20][8]);                                                                       


#include "matrices.h"



int main() {
    float A[M][K];
    float B[K][N];
    float C[M][N];

    // Initialize A and B with some values
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < K; ++j)
            A[i][j] = (float)(i + j);

    for (int i = 0; i < K; ++i)
        for (int j = 0; j < N; ++j)
            B[i][j] = (float)(i - j);

    // Initialize C to zero
    memset(C, 0, sizeof(C));

    // Blocked matrix multiplication
    for (int i = 0; i < M; i += 4) {
        for (int j = 0; j < N; j += 5) {
            float c_block[4][5] = {0}; // Temporary block result
            for (int k = 0; k < K; k += 5) {
                //float a_block[4][5], b_block[5][5];
                float a_block[20], b_block[25];

                // Copy A block
                for (int ii = 0; ii < 4; ++ii)
                    for (int kk = 0; kk < 5; ++kk)
                        a_block[ii*5+kk] = A[i + ii][k + kk];

                // Copy B block
                for (int kk = 0; kk < 5; ++kk)
                    for (int jj = 0; jj < 5; ++jj)
                        b_block[kk*5+jj] = B[k + kk][j + jj];

                // Accumulate into temporary block
                float c_partial[4][5];
                deepmatmul(a_block, b_block, c_partial);
                //deepmatmul3(a_block, b_block, c_partial, hA_mas, hA_menos, hB_mas, hB_menos, hC_mas, hC_menos);
                for (int ii = 0; ii < 4; ++ii)
                    for (int jj = 0; jj < 5; ++jj)
                        c_block[ii][jj] += c_partial[ii][jj];
            }

            // Copy block result into global C
            for (int ii = 0; ii < 4; ++ii)
                for (int jj = 0; jj < 5; ++jj)
                    C[i + ii][j + jj] = c_block[ii][jj];
        }
    }

    // Print part of result to verify
    printf("C[0][0] = %f\n", C[0][0]);
    printf("C[M-1][N-1] = %f\n", C[M-1][N-1]);


    for (int ii = 0; ii < 4; ++ii){
        for (int jj = 0; jj < 5; ++jj)
            printf("%f\t", C[ii][jj]);
        printf("\n");
    }
    return 0;
}
