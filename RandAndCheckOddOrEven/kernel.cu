/*
* # problema:
*
*	impar ou par?
*	-escalorar recursos de processamento em paralelo via dispositivo de GPU.
*	-gerar valores aleatórios entre ]10, 1000[, para processar (par ou ímpar).
*/

#include <stdio.h>
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <time.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <curand_kernel.h>

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)

std::chrono::steady_clock::time_point time_begin;
std::chrono::steady_clock::time_point time_end;
int time_result;

//Devide
//https://docs.nvidia.com/cuda/curand/device-api-overview.html#device-api-overview
__global__ void setup_kernel(int seed, curandState* state) {
	int id = threadIdx.x + blockIdx.x * blockDim.x;

	/* Each thread gets same seed, a different sequence
	   number, no offset */
	curand_init(seed, id, 0, &state[id]);
}


__global__ void k_randgenerate(int threadCount, int seed, curandState* state, int* sequence) {
	int dim = blockDim.x;
	int id = threadIdx.x;

	while (id < threadCount) {
		curand_init(seed, id, 0, &state[id]);
		curandState localState = state[id];

		sequence[id] = curand(&localState) % 989 + 11; // it looks ok
		id += dim;
	}
}

__global__ void new_k_randgenerate(int threadCount, int seed, curandState* state, int* sequence) {
	int dim = blockDim.x;
	int id = threadIdx.x;

	while (id < threadCount) {
		curandState localState = state[id];

		sequence[id] = curand(&localState) % 989 + 11; // it looks ok
		id += dim;

		state[id] = localState;
	}
}

__global__ void even_odd_check(int threadCount, int* sequence, int* result) {
	int dim = blockDim.x;
	int id = threadIdx.x;

	while (id < threadCount) {
		result[id] = sequence[id] % 2;
		id += dim;
	}
}

__global__ void k_even_odd_check(int threadCount, int seed, curandState* state, int* sequence, int* result) {
	int dim = blockDim.x;
	int id = threadIdx.x;

	while (id < threadCount) {
		curand_init(seed, id, 0, &state[id]); // !!! custo consideravel
		curandState localState = state[id];

		sequence[id] = curand(&localState) % 989 + 11; // it looks ok
		result[id] = sequence[id] % 2;
		id += dim;
	}
}

__global__ void new_k_even_odd_check(int threadCount, int seed, curandState* state, int* sequence, int* result) {
	int dim = blockDim.x;
	int id = threadIdx.x;

	while (id < threadCount) {
		curandState localState = state[id];

		sequence[id] = curand(&localState) % 989 + 11; // it looks ok
		result[id] = sequence[id] % 2;
		id += dim;

		state[id] = localState;
	}
}


// host
void k_randgenerate(int* result, int range) {
	for (int i = 0; i < range; i++) 
		result[i] = rand() % 989 + 11;
}

void even_odd_check(int* sequence, int* result, int range) {
	for (int i = 0; i < range; i++)
		result[i] = sequence[i] % 2;
}

void present_result(int* sequence, int* result, int range) {
	for (int i = 0; i < range; i++) {
		if (i < 10 || i >(range - 10))
			printf("\nvalue: %d\t check(even|odd - 0|1): %d", sequence[i], result[i]);
		else printf(".");

		// check correct delimitantion of rand 
		if (sequence[i] <= 10 || sequence[i] >= 1000) {
			printf("%d\n", sequence[i]);
			printf("Match a value far from delimitations!!!\n");
			break;
		}
	}
}

void present_time_result() {
	time_result = std::chrono::duration_cast<std::chrono::nanoseconds>(time_end - time_begin).count();
	printf("\n.started at: %dns \t.finished at: %dns \t.costs (time): %dns\n", time_begin, time_end, time_result);
}

void mark_time(std::chrono::steady_clock::time_point* time) {
	*time = std::chrono::steady_clock::now();
}

void mark_begin_clock() {
	mark_time(&time_begin);
}

void mark_end_clock() {
	mark_time(&time_end);
}

int main() {

	int size = 8192;
	int threadMaxSize = 1024;
	int blockCount = size > threadMaxSize? (size + 1) / threadMaxSize : 1;
	int threadCount = blockCount > 1? threadMaxSize : size;

	int *h_sequence, *d_sequence, *h_result, *d_result;
	curandState *d_state;
	time_t seed;

	int device;
	struct cudaDeviceProp properties;

	/* check for double precision support */
	CUDA_CALL(cudaGetDevice(&device));
	CUDA_CALL(cudaGetDeviceProperties(&properties, device));

	h_sequence = (int*)malloc(size * sizeof(int));
	h_result = (int*)malloc(size * sizeof(int));

	CUDA_CALL(cudaMalloc((void**)&d_sequence, size * sizeof(int)));
	CUDA_CALL(cudaMalloc((void**)&d_result, size * sizeof(int)));
	CUDA_CALL(cudaMalloc((void**)&d_state, sizeof(curandState)));


	// setup time and kernel for curand
	std::cout << ".quantidade de blocos: " << blockCount << std::endl;
	std::cout << ".quantidade de threads: " << threadCount << std::endl;
	std::cout << ".device state: " << d_state << std::endl;
	seed = time(NULL);
	setup_kernel << <blockCount, threadCount>> > (seed, d_state);

	// processing in device
	printf("\n>processing in \"device\" (two factors) -> rand and then check:");
	mark_begin_clock();

	k_randgenerate<<<blockCount, threadCount>>>(size, seed, d_state, d_sequence);
	cudaDeviceSynchronize();
	cudaMemcpy(h_sequence, d_sequence, size * sizeof(int), cudaMemcpyDeviceToHost);

	even_odd_check<<<blockCount, threadCount >>>(size, d_sequence, d_result);
	cudaDeviceSynchronize();
	cudaMemcpy(h_result, d_result, size * sizeof(int), cudaMemcpyDeviceToHost);

	mark_end_clock();
	present_time_result();
	present_result(h_sequence, h_result, size);

	// processing in device
	printf("\n>processing in \"device\" (two factors - new) -> rand and then check:");
	mark_begin_clock();

	new_k_randgenerate << <blockCount, threadCount >> > (size, seed, d_state, d_sequence);
	cudaDeviceSynchronize();
	cudaMemcpy(h_sequence, d_sequence, size * sizeof(int), cudaMemcpyDeviceToHost);

	even_odd_check << <blockCount, threadCount >> > (size, d_sequence, d_result);
	cudaDeviceSynchronize();
	cudaMemcpy(h_result, d_result, size * sizeof(int), cudaMemcpyDeviceToHost);

	mark_end_clock();
	present_time_result();
	present_result(h_sequence, h_result, size);

	//time_cost = time_count_end - time_count_begin;
	//printf("\n.started at: %ds \t.finished at: %ds \t.costs (time): %ds\n", time_count_begin, time_count_end, time_cost);
	//printf("\n.started at: %ds \t.finished at: %ds \t.costs (time): %ds\n", time_begin, time_end, time_result);

	//processing in device
	printf("\n>processing in \"device\" -> rand and check):");
	mark_begin_clock();

	k_even_odd_check<<<blockCount, threadCount >>>(size, seed, d_state, d_sequence, d_result);
	cudaDeviceSynchronize();
	cudaMemcpy(h_sequence, d_sequence, size * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_result, d_result, size * sizeof(int), cudaMemcpyDeviceToHost);

	mark_end_clock();
	present_time_result();
	present_result(h_sequence, h_result, size);

	printf("\n>processing in \"device\" (new) -> rand and check:");
	mark_begin_clock();

	new_k_even_odd_check << <blockCount, threadCount >> > (size, seed, d_state, d_sequence, d_result);
	cudaDeviceSynchronize();
	cudaMemcpy(h_sequence, d_sequence, size * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_result, d_result, size * sizeof(int), cudaMemcpyDeviceToHost);

	mark_end_clock();
	present_time_result();
	present_result(h_sequence, h_result, size);


	//processing in host
	printf("\n>processing in \"host\" (two factors) -> rand and then check):");
	mark_begin_clock();

	k_randgenerate(h_sequence, size);
	even_odd_check(h_sequence, h_result, size);

	mark_end_clock();
	present_time_result();
	present_result(h_sequence, h_result, size);

	//free(h_sequence);
	//free(h_result);
	cudaFree(d_sequence);
	cudaFree(d_result);
	cudaFree(d_state);

	return 0;
}

