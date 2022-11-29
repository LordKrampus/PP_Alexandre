#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

__global__ void cuda_invertToGrayscale(unsigned char* image, unsigned char* result, int size, int start, int channels, int* mark_step, int step_dependence, int step_value)
{
    int index = start + (threadIdx.x + blockIdx.x * blockDim.x) * channels;
    if (index > start + size) return;
    while (*mark_step < step_dependence);

    float RGBFactor = image[index + 2] * .299f + image[index + 1] * .587f + image[index + 0] * .114f;
    for (int i = 0; i < channels; i++) {
        result[index + i] = RGBFactor;
    }

    *mark_step = step_value;
}


__global__ void cuda_invertColors(unsigned char* image, unsigned char* result, int size, int start, int channels, int* mark_step, int step_dependence, int step_value)
{
    int index = start + (threadIdx.x + blockIdx.x * blockDim.x) * channels;
    if (index > start + size) return;
    while (*mark_step < step_dependence);

    float RGBFactor = 255;
    for (int i = 0; i < channels; i++) {
        result[index + i] = RGBFactor - image[index + i];
    }

    *mark_step = step_value;
}

__global__ void cuda_thresholding(unsigned char* image, unsigned char* result, int size, int start, int channels, int thresh, int* mark_step, int step_dependence, int step_value)
{
    int index = start + (threadIdx.x + blockIdx.x * blockDim.x) * channels;
    if (index > start + size) return;
    while (*mark_step < step_dependence);

    float RGBFactor = 0;
    for (int i = 0; i < channels; i++) {
        RGBFactor += image[index + i];
    }

    RGBFactor /= channels;

    if (RGBFactor < thresh)
        RGBFactor = 0;
    else
        RGBFactor = 255;

    for (int i = 0; i < channels; i++) {
        result[index + i] = RGBFactor;
    }

    *mark_step = step_value;
}

/*
void Kernel_InvertToGrayscale(unsigned char* d_image, int size, int channels, int blockCount, int threadCount) {
    cuda_invertToGrayscale <<<blockCount, threadCount >> > (d_image, size, channels);
}

void Kernel_InvertColors(unsigned char* d_image, int size, int channels, int blockCount, int threadCount) {
    cuda_invertColors << <blockCount, threadCount >> > (d_image, size, channels);
}

void Kernel_Thresholding(unsigned char* d_image, int size, int channels, int thresh, int blockCount, int threadCount) {
    cuda_thresholding << <blockCount, threadCount >> > (d_image, size, channels, thresh);
}
*/


void Run_Kernel(unsigned char** d_image, int slice_size, int start_index, int channels, int thresh, int* mark_step, int block_count, int thread_count) {
    cuda_invertToGrayscale << <block_count, thread_count >> > (d_image[0], d_image[1], slice_size, start_index, channels, mark_step, 0, 1);
    //cudaDeviceSynchronize();
    //*mark_step += 1;

    cuda_invertColors << <block_count, thread_count >> > (d_image[1], d_image[2], slice_size, start_index, channels, mark_step, 1, 2);
    //cudaDeviceSynchronize();
    //*mark_step += 1;

    cuda_thresholding << <block_count, thread_count >> > (d_image[2], d_image[3], slice_size, start_index, channels, thresh, mark_step, 2, 3);
    //cudaDeviceSynchronize();
}

/*
namespace image_manipulation {
    class Master {
    private:
        //Kernel* kernel;
        unsigned char* d_image;
        dim3 grid_image;
        int i_size;
        int channels;
        int thresh;

        void configImageSize(int width, int height, int channels, int thresh) {
            this->i_size = width * height * channels;
            this->channels = channels;
            this->thresh = thresh;
        }

        void configDeviceMem(unsigned char* h_image) {
            cudaMalloc((void**)&this->d_image, this->i_size);
            cudaMemcpy(this->d_image, h_image, this->i_size, cudaMemcpyHostToDevice);
        }

        void configGridImage(int width, int height) {
            this->grid_image = dim3(width, height);
        }

        void synchDeviceToHost(unsigned char* h_image) {
            cudaDeviceSynchronize();
            cudaMemcpy(h_image, this->d_image, this->i_size, cudaMemcpyDeviceToHost);
        }
    public:
        Master() {
            //this->kernel = new Kernel();
            this->d_image = NULL;
            this->grid_image = NULL;
            this->i_size = 0;
        }

        ~Master() {
            cudaFree(d_image);
        }

        void configDevice(unsigned char* sourceImage, int width, int height, int channels, int thresh) {
            this->configImageSize(width, height, channels, thresh);
            this->configDeviceMem(sourceImage);
            this->configGridImage(width, height);
        }

        void Kernel_RunRoutine(unsigned char* sourceImage);
    };
}


void image_manipulation::Master::Kernel_RunRoutine(unsigned char* sourceImage) {
    manipulation01(this->grid_image, 1, this->d_image, sourceImage, this->channels);
    this->synchDeviceToHost(sourceImage);

    manipulation02(this->grid_image, 1, this->d_image, sourceImage, this->channels);
    this->synchDeviceToHost(sourceImage);

    manipulation03(this->grid_image, 1, this->d_image, sourceImage, this->channels, this->thresh);
    this->synchDeviceToHost(sourceImage);
}
*/




/*
void Kernel_InvertToGrayscale(unsigned char* sourceImage, int width, int height, int channels) {
    unsigned char* inverseImage = NULL;
    int imageSize = width * height * channels;
    //int maxThreadSize = 1024;
    //int countBlocks = width * height > maxThreadSize ? (width * height / maxThreadSize) + 1: 1;
    //int countThreads = width * height > maxThreadSize ? maxThreadSize : width * height;

    //::cout << imageSize << std::endl;
    //std::cout << countBlocks << std::endl;
    //std::cout << countThreads << std::endl;

    //transfereir endereçamento
    cudaMalloc((void**)&inverseImage, imageSize);
    cudaMemcpy(inverseImage, sourceImage, imageSize, cudaMemcpyHostToDevice);

    //executar o Kernel
    dim3 Grid_Image(width, height);
    //std::cout << Grid_Image.x << std::endl;
    //std::cout << Grid_Image.y << std::endl;
    //int maxThreadSize = 1024;
    //int countBlocks = width * height > maxThreadSize ? (width * height / maxThreadSize) + 1: 1;
    //int countThreads = width * height > maxThreadSize ? maxThreadSize : width * height;
    cuda_invertToGrayscale<<<Grid_Image, 1 >>> (inverseImage, channels);
    cudaDeviceSynchronize();
    cudaMemcpy(sourceImage, inverseImage, imageSize, cudaMemcpyDeviceToHost);

    //liberar Kernel
    cudaFree(inverseImage);
}


void Kernel_InvertColors(unsigned char* sourceImage, int width, int height, int channels) {
    unsigned char* inverseImage = NULL;
    int imageSize = width * height * channels;

    //transfereir endereçamento
    cudaMalloc((void**)&inverseImage, imageSize);
    cudaMemcpy(inverseImage, sourceImage, imageSize, cudaMemcpyHostToDevice);

    //executar o Kernel
    dim3 Grid_Image(width, height);
    cuda_invertColors <<<Grid_Image, 1 >>> (inverseImage, channels);
    cudaDeviceSynchronize();

    //retornar endereçamento
    cudaMemcpy(sourceImage, inverseImage, imageSize, cudaMemcpyDeviceToHost);

    //liberar Kernel
    cudaFree(inverseImage);
}


void Kernel_Thresholding(unsigned char* sourceImage, int width, int height, int channels, int thresh) {
    unsigned char* inverseImage = NULL;
    int imageSize = width * height * channels;

    //transfereir endereçamento
    cudaMalloc((void**)&inverseImage, imageSize);
    cudaMemcpy(inverseImage, sourceImage, imageSize, cudaMemcpyHostToDevice);

    //executar o Kernel
    dim3 Grid_Image(width, height);
    cuda_thresholding <<<Grid_Image, 1 >>> (inverseImage, channels, thresh);
    cudaDeviceSynchronize();

    //retornar endereçamento
    cudaMemcpy(sourceImage, inverseImage, imageSize, cudaMemcpyDeviceToHost);

    //liberar Kernel
    cudaFree(inverseImage);
}
*/


/*
__global__ void cuda_invertToGrayscale(unsigned char* image, int channels)
{
    int x = blockIdx.x;
    int y = blockIdx.y;

    int grid = (x + y * gridDim.x) * channels;

    float RGBFactor = image[grid + 2] * .299f + image[grid + 1] * .587f + image[grid + 0] * .114f;

    for (int i = 0; i < channels; i++) {
        image[grid + i] = RGBFactor;
    }
}


__global__ void cuda_invertColors(unsigned char* image, int channels)
{
    int x = blockIdx.x;
    int y = blockIdx.y;

    int grid = (x + y * gridDim.x) * channels;

    float RGBFactor = 255;
    for (int i = 0; i < channels; i++) {
        image[grid + i] = RGBFactor - image[grid + i];
    }
}

__global__ void cuda_thresholding(unsigned char* image, int channels, int thresh)
{
    int x = blockIdx.x;
    int y = blockIdx.y;

    int grid = (x + y * gridDim.x) * channels;

    float RGBFactor = 0;
    for (int i = 0; i < channels; i++) {
        RGBFactor += image[grid + i];
    }

    RGBFactor /= channels;

    if (RGBFactor < thresh) {
        for (int i = 0; i < channels; i++) {
            image[grid + i] = 0;
        }
    }
    else {
        for (int i = 0; i < channels; i++) {
            image[grid + i] = 255;
        }
    }
}

*/