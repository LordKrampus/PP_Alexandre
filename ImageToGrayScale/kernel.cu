
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void cuda_invertToGrayscale(unsigned char* image, int channels)
{
    int x = blockIdx.x;
    int y = blockIdx.y;

    int grid = (x + y * gridDim.x) * channels;
    float RGBAFactor;
    if (channels >= 3) {
        RGBAFactor = image[grid + 2] * .299f + image[grid + 1] * .587f + image[grid + 0] * .114f;
    }

    for (int i = 0; i < channels; i++) {
        image[i + grid] = RGBAFactor;
    }
}

void InvertToGrayscale(unsigned char* sourceImage, int width, int height, int channels) {
    unsigned char* inverseImage = NULL;
    int imageSize = width * height * channels;

    //transfereir endereçamento
    cudaMalloc((void**)&inverseImage, imageSize);
    cudaMemcpy(inverseImage, sourceImage, imageSize, cudaMemcpyHostToDevice);

    //executar o Kernel
    dim3 Grid_Image(width, height);
    cuda_invertToGrayscale<<<Grid_Image, 1>>>(inverseImage, channels);

    //retornar endereçamento
    cudaMemcpy(sourceImage, inverseImage, imageSize, cudaMemcpyDeviceToHost);

    //liberar Kernel
    cudaFree(inverseImage);
}
