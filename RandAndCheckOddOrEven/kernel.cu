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
		curand_init(seed, id, 0, &state[id]);
		curandState localState = state[id];

		sequence[id] = curand(&localState) % 989 + 11; // it looks ok
		result[id] = sequence[id] % 2;
		id += dim;
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
	int length = 8192;

	int *h_sequence, *d_sequence, *h_result, *d_result;
	curandState *d_state;

	h_sequence = (int*)malloc(length * sizeof(int));
	h_result = (int*)malloc(length * sizeof(int));

	CUDA_CALL(cudaMalloc((void**)&d_sequence, length * sizeof(int)));
	CUDA_CALL(cudaMalloc((void**)&d_result, length * sizeof(int)));
	CUDA_CALL(cudaMalloc((void**)&d_state, sizeof(curandState)));

	time_t seed;
	//time_t time_count_end;
	//int time_cost;

	// processing in device
	printf("\n>processing in \"device\" (two factors -> rand and then check):");
	seed = time(NULL);
	mark_begin_clock();

	k_randgenerate<<<8, 1024>>>(length, seed, d_state, d_sequence);
	cudaDeviceSynchronize();
	cudaMemcpy(h_sequence, d_sequence, length * sizeof(int), cudaMemcpyDeviceToHost);

	even_odd_check<<<8, 1024>>>(length, d_sequence, d_result);
	cudaDeviceSynchronize();
	cudaMemcpy(h_result, d_result, length * sizeof(int), cudaMemcpyDeviceToHost);

	mark_end_clock();
	//time_count_end = time(NULL);
	present_result(h_sequence, h_result, length);

	//time_cost = time_count_end - time_count_begin;
	//printf("\n.started at: %ds \t.finished at: %ds \t.costs (time): %ds\n", time_count_begin, time_count_end, time_cost);
	//printf("\n.started at: %ds \t.finished at: %ds \t.costs (time): %ds\n", time_begin, time_end, time_result);
	present_time_result();

	//processing in device
	printf("\n>processing in \"device\" (one factor) -> rand and check):");
	//time_count_begin = time(NULL);
	mark_begin_clock();

	k_even_odd_check<<<8, 1024 >>>(length, seed, d_state, d_sequence, d_result);
	cudaDeviceSynchronize();
	cudaMemcpy(h_sequence, d_sequence, length * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_result, d_result, length * sizeof(int), cudaMemcpyDeviceToHost);

	mark_end_clock();
	//time_count_end = time(NULL);
	present_result(h_sequence, h_result, length);

	//time_cost = time_count_end - time_count_begin;
	//printf("\n.started at: %ds \t.finished at: %ds \t.costs (time): %ds\n", time_count_begin, time_count_end, time_cost);
	//printf("\n.started at: %ds \t.finished at: %ds \t.costs (time): %ds\n", time_begin, time_end, time_result);
	present_time_result();


	//processing in host
	printf("\n>processing in \"host\" (two factors -> rand and then check):");
	//time_count_begin = time(NULL);
	mark_begin_clock();

	k_randgenerate(h_sequence, length);
	even_odd_check(h_sequence, h_result, length);

	mark_end_clock();
	//time_count_end = time(NULL);
	present_result(h_sequence, h_result, length);

	//time_cost = time_count_end - time_count_begin;
	//printf("\n.started at: %ds \t.finished at: %ds \t.costs (time): %ds\n", time_count_begin, time_count_end, time_cost);
	present_time_result();


	//free(h_sequence);
	//free(h_result);
	cudaFree(d_sequence);
	cudaFree(d_result);
	cudaFree(d_state);

	return 0;
}

