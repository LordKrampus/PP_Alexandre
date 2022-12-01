#include <stdio.h>

void Core_InvertToGrayscale(unsigned char* sourceImage, int width, int height, int channels) {
	int size = width * height * channels;

	float RGBFactor;
	for (int i = 0; i < size; i += channels) {
		RGBFactor = sourceImage[i + 2] * .299f + sourceImage[i + 1] * .587f + sourceImage[i + 0] * .114f;

		for (int j = 0; j < channels; j++)
			sourceImage[i + j] = RGBFactor;
	}
}

void Core_InvertColors(unsigned char* sourceImage, int width, int height, int channels) {
	int size = width * height * channels;

	float RGBFactor = 255;
	for (int i = 0; i < size; i += channels) {
		for (int j = 0; j < channels; j++)
			sourceImage[i + j] = RGBFactor - sourceImage[i + j];
	}
}

void Core_Thresholding(unsigned char* sourceImage, int width, int height, int channels, int thresh) {
	int size = width * height * channels;

	float RGBFactor;
	for (int i = 0; i < size; i += channels) {
		RGBFactor = 0;

		// média
		for (int j = 0; j < channels; j++)
			RGBFactor += sourceImage[i + j];

		RGBFactor /= channels;

		if (RGBFactor < thresh)
			RGBFactor = 0;
		else
			RGBFactor = 255;

		for (int j = 0; j < channels; j++)
			sourceImage[i + j] = RGBFactor;
	}
}
