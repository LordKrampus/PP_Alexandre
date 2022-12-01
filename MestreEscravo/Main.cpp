#include <stdio.h>
#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "Methods.h"
#include "Device.h"
#include "Utilities.h"
//#include "Kernel.cu"

const char* IMAGE_PATH_FROM = ".\\imageBase\\Read\\";
const char* IMAGE_PATH_TO = ".\\imageBase\\Write\\";
const char* DEFAULT_IMAGE_NAME_FROM = "SourceImage.png";
const char* DEFAULT_IMAGE_NAME_TO = "ProcessedImage.png";
int DEFAULT_THRESH = 150;


void ShowImage(cv::Mat image, std::string header) {
    cv::imshow(header, image);
    cv::waitKey(0);
}

int main(int argc, char* argv[]) {
    if (argc < 2) return 0;
    //time(NULL);
    std::cout << argv[0] << std::endl;

    std::string imageFrom;
    std::string imageTo;
    std:int thresh; 

    ClockTime* ct = new ClockTime();
    Manager* mt = new Manager(strtol(argv[1], NULL, 10));

    imageFrom = IMAGE_PATH_FROM;
    imageTo = IMAGE_PATH_TO;

    if (argc > 2) {
        imageFrom += argv[2];

        if (argc > 3) {
            imageTo += argv[3];

            if (argc > 4)
                //https://stackoverflow.com/questions/9748393/how-can-i-get-argv-as-int
                thresh = strtol(argv[4], NULL, 10);
            else
                goto DEFAULT_THRESH;

            goto CONTINUE;
        }
        else goto DEFAULT_NAME_TO;
    }
    else goto DEFAULT_NAME_FROM;

    DEFAULT_NAME_FROM: {
    imageFrom += DEFAULT_IMAGE_NAME_FROM;
    goto DEFAULT_NAME_TO;
    }

    DEFAULT_NAME_TO: {
    imageTo += DEFAULT_IMAGE_NAME_TO;
    goto DEFAULT_THRESH;
    }

    DEFAULT_THRESH: {
    thresh = DEFAULT_THRESH;
    goto CONTINUE;
    }

    CONTINUE: {}

    //Processar na CPU - Serial
    //*
    std::cout << "Carregando Imagem:\n" << ".from: \"" << imageFrom << "\"\n.to: \"" << imageTo << "\"" << std::endl;
    cv::Mat inputImage = cv::imread(imageFrom);
    std::cout << "Informacoes da imagem:\n" << "." << inputImage.rows << " rows\t." << inputImage.cols << " collumns" << std::endl;

    if (inputImage.empty()) {
        std::cout << ".exception: can't read the imagem (" << imageFrom << ")." << std::endl;
        return 0;;
    }
    cv::imshow("entrada: " + (std::string)DEFAULT_IMAGE_NAME_FROM, inputImage);
    cv::waitKey(0);

    ct->mark_begin_clock();
    Core_InvertToGrayscale(inputImage.data, inputImage.rows, inputImage.cols, inputImage.channels());
    ct->mark_end_clock();
    ct->present_time_stamp("\n (processamento)\n");
    //imwrite("core_0_" + imageTo, inputImage);
    ShowImage(inputImage, "core - Invert To Grayscale");

    ct->mark_begin_clock();
    Core_InvertColors(inputImage.data, inputImage.rows, inputImage.cols, inputImage.channels());
    ct->mark_end_clock();
    ct->present_time_stamp("\n (processamento)\n");
    //imwrite("core_1_" + imageTo, inputImage);
    ShowImage(inputImage, "core - Invert Colors");

    ct->mark_begin_clock();
    Core_Thresholding(inputImage.data, inputImage.rows, inputImage.cols, inputImage.channels(), thresh);
    ct->mark_end_clock();
    ct->present_time_stamp("\n (processamento)\n");
    //imwrite("core_2_" + imageTo, inputImage);
    ShowImage(inputImage, "core - Thresholding (" + std::to_string(thresh) + ")");

    ct->present_time_buffer(" (resultado final)\n");
    imwrite("core_" + imageTo, inputImage);
    //*/

    ct->reset_time_buffer();

    //Processar no kernel - Mestre Escravo
    std::cout << "Carregando Imagem:\n" << ".from: \"" << imageFrom << "\"\n.to: \"" << imageTo << "\"" << std::endl;
    //cv::Mat inputImage = cv::imread(imageFrom);
    inputImage = cv::imread(imageFrom);
    std::cout << "Informacoes da imagem:\n" << "." << inputImage.rows << " rows\t." << inputImage.cols << " collumns" << std::endl;

    ct->mark_begin_clock();
    mt->configDevice(inputImage.data, inputImage.rows, inputImage.cols, inputImage.channels(), thresh);
    ct->mark_end_clock();
    ct->present_time_stamp(" \n(configuracao do dispositivo)\n");


    int last_step = -1, count_step;
    ct->mark_begin_clock();
    mt->RunRoutine();
    ct->mark_end_clock();
    ct->present_time_stamp(" \n(disparo de processamento no dispositivo)\n");

    ct->mark_begin_clock();
    // sincroniza até a imagem processada
    for (int i = 0; i < 4; ) {
        count_step = mt->checkStep();

        if (count_step > last_step){
            ct->mark_end_clock();
            std::cout << count_step << std::endl;
            ct->present_time_stamp(" \n(processamento e verificacao da flag)\n");

            ct->mark_begin_clock();
            mt->synchDeviceToHost(inputImage.data, i);
            ct->mark_end_clock();
            ct->present_time_stamp(" \n(barramento de resultado)\n");

            ShowImage(inputImage, "kernel -  processing (" + std::to_string(thresh) + ")");
            last_step = i;
            i++;

            ct->mark_begin_clock();
        }
        else 
            count_step = 0;
    }
    //*/

    mt->~Manager();
    ct->mark_end_clock();
    ct->present_time_stamp(" \n(liberando memoria do dispositivo)\n");


    ct->present_time_buffer(" \n(resultado final)\n");
    ShowImage(inputImage, "kernel - Thresholding (" + std::to_string(thresh) + ")");
    imwrite(imageTo, inputImage);


    free(ct);
    free(mt);

    return 0;
}


