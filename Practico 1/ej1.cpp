#include <iostream>
#include <iterator>
#include <random>
#include <algorithm>

int main() {
    long size = 1024 * 1024 * 500; // 500MB

    // Reservar memoria para el arreglo de caracteres
    char    *array   = new char[size];
    int     *indexes = new int[size];
    clock_t start;
    clock_t end;
    float   seconds;

    for (int i = 0; i < size; i++) {
        indexes[i] = i;
    }

    start = clock();
        for (int i = 0; i < size; i++) {
            array[indexes[i]] = array[indexes[i]];
        }
    end = clock();

    seconds = (float)(end - start) / CLOCKS_PER_SEC;
    std::cout << "Tiempo de ejecución secuencial: " << seconds << " segundos" << std::endl;

    std::shuffle(indexes, indexes + size, std::mt19937{std::random_device{}()});

    start = clock();
        for (int i = 0; i < size; i++) {
            array[indexes[i]] = array[indexes[i]];
        }
    end = clock();
    
    seconds = (float)(end - start) / CLOCKS_PER_SEC;
    std::cout << "Tiempo de ejecución random: " << seconds << " segundos" << std::endl;
}