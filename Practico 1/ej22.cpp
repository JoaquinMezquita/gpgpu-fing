#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>

//-----------------------------------------------------------------
// Función para multiplicar matrices SIN BLOQUES
// A es de tamaño N x M, B es de tamaño M x K, C es de tamaño N x K
//-----------------------------------------------------------------
void matrixMultNoBlocking(const float *A, const float *B, float *C,
                          int N, int M, int K)
{
    // Inicializar C en cero
    for (int i = 0; i < N * K; i++)
    {
        C[i] = 0.0f;
    }

    // Triple bucle clásico
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < K; j++)
        {
            float sum = 0.0f;
            for (int r = 0; r < M; r++)
            {
                sum += A[i * M + r] * B[r * K + j];
            }
            C[i * K + j] = sum;
        }
    }
}

//-----------------------------------------------------------------
// Función para multiplicar matrices CON BLOQUES
// A es de tamaño N x M, B es de tamaño M x K, C es de tamaño N x K
//-----------------------------------------------------------------
void matrixMultBlocked(const float *A, const float *B, float *C,
                       int N, int M, int K, int BLOCK_SIZE)
{
    // Inicializar C en cero
    for (int i = 0; i < N * K; i++)
    {
        C[i] = 0.0f;
    }

    // Recorremos en bloques
    for (int i0 = 0; i0 < N; i0 += BLOCK_SIZE)
    {
        for (int j0 = 0; j0 < K; j0 += BLOCK_SIZE)
        {
            for (int r0 = 0; r0 < M; r0 += BLOCK_SIZE)
            {

                // Límite efectivo del bloque (en caso de que no sea múltiplo exacto)
                int iMax = std::min(i0 + BLOCK_SIZE, N);
                int jMax = std::min(j0 + BLOCK_SIZE, K);
                int rMax = std::min(r0 + BLOCK_SIZE, M);

                // Multiplicación parcial en el bloque
                for (int i = i0; i < iMax; i++)
                {
                    for (int j = j0; j < jMax; j++)
                    {
                        float sum = 0.0f;
                        for (int r = r0; r < rMax; r++)
                        {
                            sum += A[i * M + r] * B[r * K + j];
                        }
                        C[i * K + j] += sum;
                    }
                }
            }
        }
    }
}

//-----------------------------------------------------------------
// Función auxiliar para calcular MFLOPS
//-----------------------------------------------------------------
double computeMflops(int N, int M, int K, double seconds)
{
    // Aproximadamente 2*N*M*K FLOPs (multiplicación + suma)
    double flops = 2.0 * static_cast<double>(N) * static_cast<double>(M) * static_cast<double>(K);
    // MFLOPS = FLOPs / (tiempo * 1e6)
    double mflops = (flops / seconds) / 1.0e6;
    return mflops;
}

//-----------------------------------------------------------------
// Programa principal
//-----------------------------------------------------------------
int main()
{
    // Leer N, M, K y BLOCK_SIZE desde stdin
    int N, M, K, BLOCK_SIZE;
    std::cout << "Ingrese N (filas de A): ";
    std::cin >> N;
    std::cout << "Ingrese M (columnas de A / filas de B): ";
    std::cin >> M;
    std::cout << "Ingrese K (columnas de B): ";
    std::cin >> K;
    std::cout << "Ingrese BLOCK_SIZE: ";
    std::cin >> BLOCK_SIZE;

    // Reservar memoria para A, B, C (float)
    std::vector<float> A(N * M);
    std::vector<float> B(M * K);
    std::vector<float> C1(N * K); // Para la versión sin bloque
    std::vector<float> C2(N * K); // Para la versión con bloque

    // Generador de números aleatorios en [0,1)
    std::mt19937 gen(12345); // semilla fija para reproducibilidad
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    // Inicializar A y B con valores aleatorios
    for (int i = 0; i < N * M; i++)
    {
        A[i] = dist(gen);
    }
    for (int i = 0; i < M * K; i++)
    {
        B[i] = dist(gen);
    }

    // -------------------------------------------------------------
    // 1. Multiplicación SIN BLOQUES
    // -------------------------------------------------------------
    auto startNoBlock = std::chrono::high_resolution_clock::now();

    matrixMultNoBlocking(A.data(), B.data(), C1.data(), N, M, K);

    auto endNoBlock = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedNoBlock = endNoBlock - startNoBlock;
    double timeNoBlock = elapsedNoBlock.count();
    double mflopsNoBlock = computeMflops(N, M, K, timeNoBlock);

    std::cout << "\n--- Multiplicacion de matrices (SIN BLOQUES) ---\n";
    std::cout << "Tiempo transcurrido: " << timeNoBlock << " s\n";
    std::cout << "Rendimiento: " << mflopsNoBlock << " MFLOPS\n";

    // -------------------------------------------------------------
    // 2. Multiplicación CON BLOQUES
    // -------------------------------------------------------------
    auto startBlock = std::chrono::high_resolution_clock::now();

    matrixMultBlocked(A.data(), B.data(), C2.data(), N, M, K, BLOCK_SIZE);

    auto endBlock = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedBlock = endBlock - startBlock;
    double timeBlock = elapsedBlock.count();
    double mflopsBlock = computeMflops(N, M, K, timeBlock);

    std::cout << "\n--- Multiplicacion de matrices (CON BLOQUES) ---\n";
    std::cout << "BLOCK_SIZE = " << BLOCK_SIZE << "\n";
    std::cout << "Tiempo transcurrido: " << timeBlock << " s\n";
    std::cout << "Rendimiento: " << mflopsBlock << " MFLOPS\n";

    // -------------------------------------------------------------
    // (Opcional) Verificación simple de resultados
    // -------------------------------------------------------------
    // Podríamos comprobar que C1 y C2 son similares (dentro de cierto umbral)
    // para asegurarnos de que ambas versiones producen el mismo resultado.
    double maxDiff = 0.0;
    for (int i = 0; i < N * K; i++)
    {
        double diff = std::fabs(C1[i] - C2[i]);
        if (diff > maxDiff)
        {
            maxDiff = diff;
        }
    }
    std::cout << "\nDiferencia maxima entre C1 y C2: " << maxDiff << "\n";
    if (maxDiff < 1e-5)
    {
        std::cout << "Los resultados son (casi) iguales.\n";
    }
    else
    {
        std::cout << "Atencion: Hay diferencias notables en los resultados.\n";
    }

    return 0;
}