/***********************************************************************************************************************
This file is part of "SIFT-GPU-Thesis" project.
 * Copyright (C) 2015 {Christos Papazachariou} <{christospapazachariou@gmail.com}>
 * University of Piraeus- MSc program: Advanced Information Systems ("Embedded System Technologies" branch).
 * Partial or otherwise use of this project and/or source code is permitted only for educational and academic purposes.
 * Partial or otherwise use of this project and/or source code for commercial or militay applications is prohibited.
***********************************************************************************************************************/

#ifndef _GPUSIFT_H_
#define _GPUSIFT_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

	__global__ void GPUgraykernel(unsigned char* inputRed,unsigned char* inputGreen,unsigned char* inputBlue,unsigned char* output, int inH, int inW);

	void GPUgrayscale(unsigned char* input[3], unsigned char* output, int inH, int inW);

	__global__ void GPUdownsample(unsigned char* target,
                int targetW, int targetH,
                unsigned char* src,
				int srcW, int srcH);

	void GPUbuildOB(unsigned char* img, 
             unsigned char* octaves[MAX_O], 
             int O, int S, 
			 int* octavesW, int* octavesH);

	void GPUbuildSS(unsigned char* octaves[MAX_O], 
             float* scaleSpace[MAX_O][MAX_S], 
             int O, int S, 
             int* octavesW, int* octavesH, 
             float sigmas[MAX_O][MAX_S]);

	__global__ void GPUGaussX(unsigned char* input, float* output, float* coef1d, int gR, int inH, int inW);
	__global__ void GPUGaussY(float* input, float* output, float* coef1d, int gR, int inH, int inW);
	__global__ void GPUsrcPadding(unsigned char* srcIn, unsigned char* srcPaddedOut, float* intermediate, int gR, int inH, int inW);
	
	void GPUDoG(float *dog[MAX_O][MAX_S-1], 
			 float *scaleSpace[MAX_O][MAX_S], 
			 int O, int S,
			 int* octavesW, int* octavesH);

	__global__ void GPUdogKernel(float* SSU,	//input blurred image upper scale
			float* SSL,							//input blurred image lower scale
			float* dogout,						//output DoG image	
			int inW, int inH);					//input images dims

	void showFreeBytes();

	void GPUextreme(pointList** keyHead, int* keyN,
				 float* dog[MAX_O][MAX_S-1], 
				 int O, int S,
				 int* octavesW, int* octavesH);

	__global__ void GPUcheckExtrema(float* inself, float *inup, float* indown,int inW, int inH, float inradius, float inpeakThres, int* outflagExtrema);
	__global__ void GPUcheckEdge(float* inself,int inW,int inH,int* inflagExtrema, int* outflagEdge);
		void TESTcheckEdge(float* inself,int inW,int inH,int* inflagExtrema, int* outflagEdge);
			__global__ void GPUcheckEE(float* inself, float *inup, float* indown,int inW,int inH, float inradius,float inpeakThres,int* outflagEdge);
	void siftkeygen(int* inflagEdge, int* inflagExtrema, int inW, int inH, int i, int j,pointList** keyHead, int* keyN);

#endif//_GPUSIFT_H_
