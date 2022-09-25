#include <stdio.h>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "Methods.h"

const char* IMAGE_PATH_FROM = ".\\imageBase\\Read\\";
const char* IMAGE_PATH_TO = ".\\imageBase\\Write\\";
const char* DEFAULT_IMAGE_NAME_FROM = "SourceImage.png";
const char* DEFAULT_IMAGE_NAME_TO = "ProcessedImage.png";

//using namespace cv;

int main(int argc, char* argv[]) {
    if (argc < 1) return 0;;
    std::cout << argv[0] << std::endl;


    std::string imageFrom;
    std::string imageTo;

    imageFrom = IMAGE_PATH_FROM;
    imageTo = IMAGE_PATH_TO;

    if (argc > 1) {
        imageFrom += argv[1];

        if (argc > 2)
            imageTo += argv[2];
        else goto DEFAULT_NAME_TO;
    }
    else goto DEFAULT_NAME_FROM;

    DEFAULT_NAME_FROM: {
        imageFrom += DEFAULT_IMAGE_NAME_FROM; 
        goto DEFAULT_NAME_TO;
    }

    DEFAULT_NAME_TO: {
        imageTo += DEFAULT_IMAGE_NAME_TO;
    }

    std::cout << "Carregando Imagem:\n" << ".from: \"" << imageFrom << "\"\n.to: \"" << imageTo << "\"" << std::endl;
    cv::Mat inputImage = cv::imread(imageFrom);
    std::cout << "Informacoes da imagem:\n" << "." << inputImage.rows << " rows\t." << inputImage.cols << " collumns" << std::endl;

    if (inputImage.empty()){
        std::cout << ".exception: can't read the imagem (" << imageFrom << ")." << std::endl;
        return 0;;
    }
    cv::imshow("entrada: " + (std::string)DEFAULT_IMAGE_NAME_FROM, inputImage);
    cv::waitKey(0);

    //chamar kernel
    InvertToGrayscale(inputImage.data, inputImage.rows, inputImage.cols, inputImage.channels());

    cv::imshow("saida: " + (std::string)DEFAULT_IMAGE_NAME_TO, inputImage);
    cv::waitKey(0);

    imwrite(imageTo, inputImage);

    return 0;
}


