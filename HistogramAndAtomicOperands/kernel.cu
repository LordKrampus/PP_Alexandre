#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Methods.h"


#define THREAD_LIMIT 1024
#define BLOCK_COUNT 1024


/*
* Retorna um valor arredondado da nota correspondendo a um indíce no histograma de notas
*/
__device__ int CuDetectRange(const float& nota) {
	float range;
	float fracao;

	fracao = nota - (int)nota;
	range = (int)nota;
	if (fracao >= 0.5f)
		range += 1;

	return range;
}

/*
* calcula os valores no histograma de notas
*/
__global__ void Kernel_CuCalcHist(int* hist, float* notas) {
	int index = threadIdx.x + blockDim.x * blockIdx.x;

	hist[CuDetectRange(notas[index])]++;
}


__global__ void Kernel_CuAtomicCalcHist(int* hist, float* notas) {
	int index = threadIdx.x + blockDim.x * blockIdx.x;

	atomicAdd(&hist[CuDetectRange(notas[index])], 1);
}


void CuCalcHistStart(float* notas, const int notas_size, int* hist, const int hist_size, int* d_hist, float* d_notas) {
	cudaMalloc((void**)&d_hist, hist_size);
	cudaMemcpy(d_hist, hist, hist_size, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&d_notas, notas_size);
	cudaMemcpy(d_notas, notas, notas_size, cudaMemcpyHostToDevice);
}

void CuCalcHistFinish(int* hist, const int hist_size, int* d_hist, float* d_notas) {
	cudaDeviceSynchronize();
	cudaMemcpy(hist, d_hist, hist_size, cudaMemcpyDeviceToHost);
	
	cudaFree(d_hist);
	cudaFree(d_notas);
}


void CuAtomicCalcHist(float* notas, const int notas_size, int* hist, const int hist_size) {
	int* d_hist;
	float* d_notas;

	cudaMalloc((void**)&d_hist, hist_size);
	cudaMemcpy(d_hist, hist, hist_size, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&d_notas, notas_size);
	cudaMemcpy(d_notas, notas, notas_size, cudaMemcpyHostToDevice);

	Kernel_CuAtomicCalcHist << <1, notas_size / sizeof(float) >> > (d_hist, d_notas);

	cudaDeviceSynchronize();
	cudaMemcpy(hist, d_hist, hist_size, cudaMemcpyDeviceToHost);

	cudaFree(d_hist);
	cudaFree(d_notas);
}

void CuCalcHist(float* notas, const int notas_size, int* hist, const int hist_size) {
	int* d_hist;
	float* d_notas;

	cudaMalloc((void**)&d_hist, hist_size);
	cudaMemcpy(d_hist, hist, hist_size, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&d_notas, notas_size);
	cudaMemcpy(d_notas, notas, notas_size, cudaMemcpyHostToDevice);

	Kernel_CuCalcHist << <1, notas_size / sizeof(float) >> > (d_hist, d_notas);

	cudaDeviceSynchronize();
	cudaMemcpy(hist, d_hist, hist_size, cudaMemcpyDeviceToHost);

	cudaFree(d_hist);
	cudaFree(d_notas);
}
