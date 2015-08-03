/***********************************************************************************************************************
This file is part of "SIFT-GPU-Thesis" project.
 * Copyright (C) 2015 {Christos Papazachariou} <{christospapazachariou@gmail.com}>
 * University of Piraeus- MSc program: Advanced Information Systems ("Embedded System Technologies" branch).
 * Partial or otherwise use of this project and/or source code is permitted only for educational and academic purposes.
 * Partial or otherwise use of this project and/or source code for commercial or militay applications is prohibited.
***********************************************************************************************************************/

#include "sift.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

extern "C" {
	#include "gpusift.h"
	#include "MYtimer.h"
}
#include <windows.h>	//needed only for console text color management

extern "C"{
	void GPUgrayscale(unsigned char* input[3], unsigned char* output, int inH, int inW){

		unsigned char* GPUinput[3];
		unsigned char* GPUoutput;
		int i=0;

		printf("GPU Grayscale conversion call...\n");

//CUDAMALLOCS
		logtime("GPUgrayscale cudaMalloc-start");
		for (i=0;i<3;i++){
			if(cudaMalloc(&GPUinput[i],sizeof(unsigned char)*inH*inW)!=cudaSuccess){
				cudaError_t error = cudaGetLastError();
				printf ("cudaMalloc Error! GPUinput\tCUDA error: %s\n",cudaGetErrorString(error));	

				cudaFree(GPUinput);
			}
		}

		if(cudaMalloc(&GPUoutput,sizeof(unsigned char)*inH*inW)!=cudaSuccess){
			cudaError_t error = cudaGetLastError();
			printf ("cudaMalloc Error! GPUoutput\tCUDA error: %s\n",cudaGetErrorString(error));	
			cudaFree(GPUoutput);
		}
		logtime("GPUgrayscale cudaMalloc-end");

//CUDAMEMCPYS TO DEV
		logtime("GPUgrayscale upload-start");
		for(i=0;i<3;i++){
			if(cudaMemcpy(GPUinput[i],input[i],sizeof(unsigned char)*inH*inW,cudaMemcpyHostToDevice)!=cudaSuccess){
				cudaError_t error = cudaGetLastError();
				printf ("cudaMemcpyHostToDevice Error! GPUinput\tCUDA error: %s\n",cudaGetErrorString(error));	
				cudaFree(GPUinput);
			}
		}
		logtime("GPUgrayscale upload-end");

		logtime("GPU Grayscale kernel-start");

			const dim3 block(32,32);	//reasonable block size (1024 threads/block) 
			const dim3 grid((inW+block.x-1)/block.x,(inH+block.x-1)/block.y);	//make sure at least 1x1 kernel

//KERNEL CALLS
		GPUgraykernel<<<grid,block>>>(GPUinput[0],GPUinput[1],GPUinput[2],GPUoutput,inH,inW);
		logtime("GPU Grayscale kernel-end");

//CUDAMEMCPYS TO HOST
		logtime("GPUgrayscale download-start");
		if(cudaMemcpy(output,GPUoutput,sizeof(unsigned char)*inH*inW,cudaMemcpyDeviceToHost)!=cudaSuccess){
			cudaError_t error = cudaGetLastError();
			printf ("cudaMemcpyDeviceToHost Error! GPUoutput\tCUDA error: %s\n",cudaGetErrorString(error));	
			cudaFree(GPUoutput);
		}
		logtime("GPUgrayscale download-end");

		printf("GPU Grayscale conversion ended.\n");
		cudaFree(GPUinput);
		cudaFree(GPUoutput);
	}
}

extern "C"{
	__global__ void GPUgraykernel(unsigned char* inputRed,unsigned char* inputGreen,unsigned char* inputBlue,unsigned char* output, int inH, int inW){
		const int x=blockIdx.x*blockDim.x+threadIdx.x;
		const int y=blockIdx.y*blockDim.y+threadIdx.y;
		float redFactor=0.2989;
		float greenFactor=0.5870;
		float blueFactor=0.1140;

		if ((x<inW) && (y<inH)){
			output[y*inW+x]=
			inputRed[y*inW+x]*redFactor+
			inputGreen[y*inW+x]*greenFactor+
			inputBlue[y*inW+x]*blueFactor;
		}
	}
}

extern "C"{
	//this function was placed  here to ease file management of the project, this .cu file was the simplest and shortest
	void showFreeBytes(){	//show GPU memory usage

		HANDLE h =GetStdHandle( STD_OUTPUT_HANDLE);
		SetConsoleTextAttribute(h,FOREGROUND_GREEN | FOREGROUND_INTENSITY);

		size_t free_bytes;
		size_t total_bytes;
		if (showmemoryflag!=0){
			cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
			if (err != cudaSuccess){
				printf("cudaMemGetInfo Error!\t%s\n",cudaGetErrorString(err));
			}else{
				printf("GPU memory: %f MB free out of %f MB total.\n",(float)((double)free_bytes/(float)(1024*1024)),(float)((double)total_bytes/(float)(1024*1024)));
			}
		}

		SetConsoleTextAttribute(h,0x07 | FOREGROUND_INTENSITY);
	}
}
