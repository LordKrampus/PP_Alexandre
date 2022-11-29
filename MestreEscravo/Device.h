#pragma once

#include <stdio.h>
#include <list>
#include <cmath>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define THREAD_MAX int(1024);

/*
void Kernel_InvertToGrayscale(unsigned char* d_image, int size, int channels, int blockCount, int threadCount);
void Kernel_InvertColors(unsigned char* d_image, int size, int channels, int blockCount, int threadCount);
void Kernel_Thresholding(unsigned char* d_image, int size, int channels, int thresh, int blockCount, int threadCount);
*/

void Run_Kernel(unsigned char** d_image, int slice_size, int start_index, int channels, int thresh, int* mark_step, int block_count, int thread_count);

class Manager {
private:

    unsigned char** d_image; // device image

    int i_size; // image size
    int s_size; // size of a slice image
    int channels;
    int thresh;

    int p_count; // size of processors
    int* p_steps; // procerrors steps (n; p_steps[i] <= s_size | i = {0, 1, 2, ...n})
    int* d_pSteps;

    int block_count;
    int thread_count;

    void configImageSize(int width, int height, int channels, int thresh) {
        this->i_size = width * height * channels;
        this->channels = channels;
        this->thresh = thresh;

        this->s_size = ceil(this->i_size / float(this->p_count));

        this->block_count = (this->s_size > 1024 ? ceil(this->s_size / 1024.f) : 1);
        this->thread_count = block_count > 1 ? 1024 : s_size;
    }

    void configDeviceMem(unsigned char* sourceImage) {
        cudaMalloc((void**)&this->d_image[0], this->i_size);
        cudaMalloc((void**)&this->d_image[1], this->i_size);
        cudaMalloc((void**)&this->d_image[2], this->i_size);
        cudaMalloc((void**)&this->d_image[3], this->i_size);

        cudaMemcpy(this->d_image[0], sourceImage, i_size, cudaMemcpyHostToDevice);
        cudaMemcpy(this->d_image[1], sourceImage, i_size, cudaMemcpyHostToDevice);
        cudaMemcpy(this->d_image[2], sourceImage, i_size, cudaMemcpyHostToDevice);
        cudaMemcpy(this->d_image[3], sourceImage, i_size, cudaMemcpyHostToDevice);
    }

    
    void ConfigProcessors(int p_count) {
        // prepara sincrono de CPU
        this->p_count = p_count;
        this->p_steps = (int*)calloc(this->p_count, sizeof(int));
        
        //experimental:
        /*
        for (int i = 1; i < this->p_count; i++)
            this->p_steps[i] = 3;
        */

        // aloca no device
        cudaMalloc((void**)&this->d_pSteps, this->p_count*sizeof(int));
        cudaMemcpy(this->d_pSteps, this->p_steps, this->p_count * sizeof(int), cudaMemcpyHostToDevice);
    }


public:
    Manager(int p_count) {
        this->ConfigProcessors(p_count);

        this->i_size = 0;
        this->s_size = 0;
        this->channels = 0;
        this->thresh = 0;

        this->block_count = 0;
        this->thread_count = 0;

        this->d_image = (unsigned char**)malloc(4*sizeof(unsigned char**));
    }

    ~Manager() {
        cudaFree(d_image);
        cudaFree(d_pSteps);
        free(p_steps);
    }


    void configDevice(unsigned char* sourceImage, int width, int height, int channels, int thresh) {
        this->configImageSize(width, height, channels, thresh);
        this->configDeviceMem(sourceImage);
    }

    void RunRoutine() 
    {
        int start_index;
        //for (int i = 0; i < 1; i++) {
        for (int i = 0; i < this->p_count; i++) {
            start_index = i * this->s_size;
            Run_Kernel(this->d_image, this->s_size, start_index, this->channels, this->thresh, &this->d_pSteps[i], this->block_count, this->thread_count);
        }
    }

    /*
    std::cout << "ceil" << ceil(this->s_size / 1024.0f) << std::endl;
    std::cout << "i_size" << this->i_size << std::endl;
    std::cout << "s_size" << this->s_size << std::endl;
    std::cout << "p_count" << this->p_count << std::endl;
    std::cout << "start" << start_index << std::endl;
    std::cout << "block_count" << block_count << std::endl;
    std::cout << "thread_count" << thread_count<< std::endl;
    std::cout << "In Routine" << this->p_steps[i] << std::endl;
    */

    int checkStep() {
        cudaMemcpy(this->p_steps, this->d_pSteps, this->p_count * sizeof(int), cudaMemcpyDeviceToHost);

        int step = this->p_steps[0];
        for (int i = 1; i < this->p_count; i++) {
            if (this->p_steps[i] < step) {
                step = this->p_steps[i];
            }
        }

        return step;
    }

    void synchDeviceToHost(unsigned char* result, int step) {

        cudaMemcpy(result, this->d_image[step], this->i_size, cudaMemcpyDeviceToHost);
    }
};
