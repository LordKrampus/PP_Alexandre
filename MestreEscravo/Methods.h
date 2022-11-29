#ifndef _InvertToGrayscale_
#define _InvertToGrayscale_
void Core_InvertToGrayscale(unsigned char* sourceImage, int width, int height, int channels);
void Core_InvertColors(unsigned char* sourceImage, int width, int height, int channels);
void Core_Thresholding(unsigned char* sourceImage, int width, int height, int channels, int thresh);
#endif
