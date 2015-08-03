#ifndef _IOOCV_H_
#define _IOOCV_H_

#include <opencv\cv.h>
#include <opencv\highgui.h> 
#include <opencv\cxcore.h>
#define SHOWDELAY 600
#define LONGDELAY 1000

void arr2ocv(unsigned char* input, IplImage* output, int inH, int inW);
void ocv2arr(IplImage* input, unsigned char* output, int inH, int inW);

void grayWrapper(IplImage* input, IplImage* output);

void saveWrapper(IplImage* input, char* filename);
void openWrapper(IplImage* openedImage, char*filename);

void openShow(unsigned char* input, int inH, int inW,char* debuglabel);

	void openShow2(unsigned char* input, int inH, int inW,char* debuglabel);

	

void openSave(unsigned char* input, int inH, int inW, char* filename);

void openShowFloat(float*input, char*windowname,int inH, int inW);

void openSaveFloat(float*input, char*filename,int inH, int inW);

void ocv2arrColor(IplImage* input, unsigned char* output[3],int outH, int outW);

#endif
