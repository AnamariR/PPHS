///
/// ... including system libraries so the program can use their functions ...
///
#include <assert.h>
#include <chrono>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <cuda.h>
using namespace std;
#define CHUNK_SIZE 1024
#define T unsigned long long int

__global__ void Fibonacci(T *a, int start) {

	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int index = i + start;

	if (i < 2 * start - 1) {
		a[index] = (a[start - 2] * a[i]) + (a[start - 1] * a[i + 1]);
	}
}

/**
* glavna funkcija
*/

int main(int argc, char *argv[]) {
	auto pocetak = chrono::steady_clock::now();
	int N = 96;

	T x[3] = { 1, 1, 2 };
	T *d_a;


	///
	/// ... allocate memory on the device ...
	///
	cudaMalloc(&d_a, N * sizeof(T));
	cudaMemcpy(d_a, x, sizeof(x), cudaMemcpyHostToDevice);

	unsigned int start = 3;

	///
	/// ... ceiling of start ...
	///
	while (start <= N / 2) {
		unsigned int num_blocks = (start - 1) / CHUNK_SIZE;
		if ((start - 1) % CHUNK_SIZE != 0) {
			num_blocks++;
		}
		Fibonacci << < num_blocks, CHUNK_SIZE >> > (d_a, start);
		start = 2 * start - 1;
	}

	///
	/// ... get the result array back ...
	///
	T b[96];
	cudaMemcpy(b, d_a, N * sizeof(T), cudaMemcpyDeviceToHost);

	///
	/// ... print result on the screen ...
	///
	for (int i = 0; i < 47; i++) {
		printf("%d:\t%lu \n", i + 1, b[i]);
	}
	auto end = chrono::steady_clock::now();
	chrono::duration<double> elapsed_seconds = end - pocetak;
	cout << "elapsed time: " << elapsed_seconds.count() << "s\n";


	///
	/// ... free the device memmory ...
	///
	cudaFree(d_a);


	return 0;
}