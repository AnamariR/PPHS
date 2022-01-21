/**
* Including system libraries so the program can use their functions
* -for programing with CUDA the important library is <cuda.h>
* -library <chrono> is just used to calculate time 
*/
#include <assert.h>
#include <chrono>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <cuda.h>
using namespace std;
/**
* CHUNK_SIZE is the size of a thread used
*/
#define CHUNK_SIZE 1024
/**
* T defines type ( it's just a shortcut so we don't have to write 
* unsigned long long int every time) 
*/
#define T unsigned long long int

/**
* __global__ tells to the compiler that the function will be executed on the GPU and is callable by the host
* 
* -parameters that are sent to the function are:
*
*	@param a - an array
*	@param start - the beginning thread
*
*
*- parameters that are used in the function 
*	@param blockDim.x - block dimension 
*	@param blockId.x - unique for each block
*	@param threadIdx.x - unique within each thread 
*
*/
__global__ void Fibonacci(T *a, int start) {

	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int index = i + start;
	/**
	*
	* @param i - how many threads we have 
	*/
	if (i < 2 * start - 1) {
		a[index] = (a[start - 2] * a[i]) + (a[start - 1] * a[i + 1]);
	}
}
/**
*  int main() - main function  
*/
int main(int argc, char *argv[]) {
	/**
	* begining - returns a time point representing the current point in time.
	*		   - this will later be used for calculating the time passed for the code to execute
	*/
	auto begining = chrono::steady_clock::now();
	int N = 96;
	/**
	* x[ ] - array that is given at the beginning with the starting values of the Fibonacci sequence
	* *d_a - pointer on the copied memory 
	* b[ ] - array for the resut
	*/
	T x[3] = { 1, 1, 2 };
	T *d_a;
	T b[96];
	/**
	* cudaMalloc - allocates N*sizeof(T) bytes of memory on the device and returns in &d_a a pointer to the allocated memory
	* cudaMemcpy - copies sizeof(x) bytes from the memory area pointed to by x to the memory area pointed to by d_a where cudaMemcpyHostToDevice
	*			   copies from host to device
	* start = 3 - number of given elements 
	*/
	cudaMalloc(&d_a, N * sizeof(T));
	cudaMemcpy(d_a, x, sizeof(x), cudaMemcpyHostToDevice);
	unsigned int start = 3;

	/**
	*
	* The loop goes from index = 3 to N/2
	* @param num_blocks - number of blocks that we use , we calculate it by dividing the number of threads that we used with the maximun number of 
	* threads per block - with the if statement we check if the number of blocks is a whole number , if it isn't we add 1 to the number of blocks 
	* 
	* -Fibonacci function is called 
	* - as the function is recursively called, each time it calculates twice as it did before so start has to multiply by 2 and we substract 1 
	* so it could start from the last calculated element 
	*/
	while (start <= N/2 ) {
		unsigned int num_blocks = (start - 1) / CHUNK_SIZE;
		if ((start - 1) % CHUNK_SIZE != 0) {
			num_blocks++;
		}
		Fibonacci << < num_blocks, CHUNK_SIZE >> > (d_a, start);
		start = 2 * start - 1;
	}
	/**
	*
	* cudaMemcpy this time copies N*sizeof(x) bytes from the memory area pointed to d_a to the memory pointed to b where
	* cudaMemcpyDeviceToHost copies from device to host 
	*
	*/
	
	cudaMemcpy(b, d_a, N * sizeof(T), cudaMemcpyDeviceToHost);

	/**
	* In the for loop we print out the elements of the Fibonacci sequence from the b[ ] array 
	*/
	for (int i = 0; i < 47; i++) {
		printf("%d:\t%lu \n", i + 1, b[i]);
	}
	/**
	* End represents the finishing time after we finished working on the GPU and printed out the sequence 
	* We calculate the duration of the work by substracting starting time from the ending time, at the end we print it 
	*/
	auto end = chrono::steady_clock::now();
	chrono::duration<double> elapsed_seconds = end - begining;
	cout << "elapsed time: " << elapsed_seconds.count() << "s\n";
	
	/**
	* System function for dealocation of memmory 
	*/
	cudaFree(d_a);

	return 0;
}