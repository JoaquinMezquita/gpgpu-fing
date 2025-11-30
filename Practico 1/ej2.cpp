#include <iostream>
#include <math.h>

double computeMflops(int N, int M, int K, double seconds) {
    // Aproximadamente 2*N*M*K FLOPs (multiplicación + suma)
    double flops = 2.0 * static_cast<double>(N) * static_cast<double>(M) * static_cast<double>(K);
    // MFLOPS = FLOPs / (tiempo * 1e6)
    double mflops = (flops / seconds) / 1.0e6;
    return mflops;
}

void matrix_mult(float **A, float *B, float *C, size_t size) {
    clock_t start;
    clock_t end;
    float seconds;

    for (size_t i = 0; i < size; i++)
        C[i] = 0;

    start = clock();
    for (size_t i = 0; i < size; i++)
        for (size_t j = 0; j < size; j++)
            C[i] += A[i][j] * B[j];
    end = clock();

    seconds = (float)(end - start) / CLOCKS_PER_SEC;
    std::cout << "Tiempo de cálculo matrices " << size << "x" << size << "(" << ((size * size) + (size * 2)) / 1024 << "KB) " << seconds << " segundos " << computeMflops(size, size, size, seconds) << " MFLOPS" << std::endl;
}

void matrix_mult_bl(float **A, float *B, float *C, size_t size, size_t TAM_BL) {
    clock_t start;
    clock_t end;
    float seconds;

    for (size_t i = 0; i < size; i++)
        C[i] = 0;

    start = clock();
    for (size_t jj = 0; jj < size; jj += TAM_BL)
        for (size_t i = 0; i < size; i++)
            for (size_t j = jj; j < jj + TAM_BL && j < size; j++)
                C[i] += A[i][j] * B[j];
    end = clock();

    seconds = (float)(end - start) / CLOCKS_PER_SEC;
    std::cout << "Tiempo de cálculo matrices " << size << "x" << size << "(" << ((size * size) + (size * 2)) / 1024 << "KB) con tamaño de bloque " << TAM_BL * sizeof(float) / 1024 << " KB: " << seconds << " segundos " << computeMflops(size, size, size, seconds) << " MFLOPS" << std::endl;
}

void matrix_mult(float **A, float **B, float **C, size_t size) {
    clock_t start;
    clock_t end;
    float seconds;

    for (size_t i = 0; i < size; i++)
        for (size_t j = 0; j < size; j++)
            C[i][j] = 0;

    start = clock();
    for (size_t f = 0; f < size; f++)
        for (size_t i = 0; i < size; i++)
            for (size_t c = 0; c < size; c++)
                C[f][c] += A[f][i] * B[i][c];

    end = clock();

    seconds = (float)(end - start) / CLOCKS_PER_SEC;
    std::cout << "Tiempo de cálculo matrices " << size << "x" << size << "(" << size * size * 3 / 1024 << "KB) " << seconds << " segundos " << computeMflops(size, size, size, seconds) << " MFLOPS" << std::endl;
}

void matrix_mult_bl(float **A, float **B, float **C, size_t size, size_t TAM_BL) {
    clock_t start;
    clock_t end;
    float seconds;

    for (size_t i = 0; i < size; i++)
        for (size_t j = 0; j < size; j++)
            C[i][j] = 0;

    start = clock();
    for (size_t bf = 0; bf < size; bf += TAM_BL)
        for (size_t bi = 0; bi < size; bi += TAM_BL)
            for (size_t bc = 0; bc < size; bc += TAM_BL)
                for (size_t f = bf; f < bf + TAM_BL && f < size; f++)
                    for (size_t i = bi; i < bi + TAM_BL && i < size; i++)
                        for (size_t c = bc; c < bc + TAM_BL && c < size; c++)
                            C[f][c] += A[f][i] * B[i][c];

    // for (size_t bf = 0; bf < size; bf += TAM_BL)
    //     for (size_t f = bf; f < bf + TAM_BL && f < size; f++)
    //         for (size_t bc = 0; bc < size; bc += TAM_BL)
    //             for (size_t c = bc; c < bc + TAM_BL && c < size; c++)
    //                 C[f][c] += A[f][c] * B[c][f];
    end = clock();

    seconds = (float)(end - start) / CLOCKS_PER_SEC;
    std::cout << "Tiempo de cálculo matrices " << size << "x" << size << "(" << size * size * 3 / 1024 << "KB) con tamaño de bloque " << TAM_BL * TAM_BL * 3 * sizeof(float) / 1024 << " KB: " << seconds << " segundos " << computeMflops(size, size, size, seconds) << " MFLOPS" << std::endl;
}

/*
Cache L1:	80 KB (per core)
Cache L2:	1.25 MB (per core)
Cache L3:	24 MB (shared)
*/
int main(int argc, char *argv[]) {
    int size, block = 0; // tamaño = 160000

    try {
        size = std::stoi(argv[1]);
        block = std::stoi(argv[2]);
    }
    catch (const std::exception &e) {
        std::cerr << "Error: Argumento inválido, ingrese un número entero." << std::endl;
        return 1;
    }

    // std::cout << "Ingrese tamaño matriz: ";
    // std::cin >> size;

    // std::cout << "Ingrese tamaño bloque: ";
    // std::cin >> block;

    float **A1 = (float **)malloc(size * sizeof(float *));
    float *B1 = (float *)malloc(size * sizeof(float *));
    float *C1 = (float *)malloc(size * sizeof(float *));

    float **A = (float **)malloc(size * sizeof(float *));
    float **B = (float **)malloc(size * sizeof(float *));
    float **C = (float **)malloc(size * sizeof(float *));

    for (size_t i = 0; i < size; ++i) {
        A1[i] = (float *)malloc(size * sizeof(float));

        A[i] = (float *)malloc(size * sizeof(float));
        B[i] = (float *)malloc(size * sizeof(float));
        C[i] = (float *)malloc(size * sizeof(float));
    }

    for (size_t i = 0; i < size; i++) {

    
        for (size_t j = 0; j < size; j++){
            A1[i][j] = rand();

            A[i][j] = rand();
            B[i][j] = rand();
        }
    }
    // matrix_mult(A1, B1, C1, size);
    // matrix_mult_bl(A1, B1, C1, size, block);

    matrix_mult(A, B, C, size);
    matrix_mult_bl(A, B, C, size, block);

    for (int i = 0; i < size; ++i) {
        free(A1[i]);

        free(A[i]);
        free(B[i]);
        free(C[i]);
    }

    free(A1);

    free(A);
    free(B);
    free(C);
}