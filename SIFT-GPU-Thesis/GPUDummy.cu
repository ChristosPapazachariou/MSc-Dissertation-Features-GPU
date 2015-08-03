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
#include "GPUDummy.h"
}


extern "C"{
	__global__ void GPUDummyKernel(unsigned char* GPUoutput, int inH, int inW){

		int y=blockIdx.x;
		int x=threadIdx.x;

		GPUoutput[x*inW+y]= (unsigned char)((int)((x*y)/256));
					
	}
}

extern "C"{
	void dummyWrapper(unsigned char* output,int inH, int inW){
		unsigned char* GPUtemp;

		if(cudaMalloc(&GPUtemp,sizeof(unsigned char)*inW*inH)!=cudaSuccess){
			printf ("cudaMalloc Error!\n");	
			getchar();
			cudaFree(GPUtemp);
		}else{
			printf ("cudaMalloc Success!\n");	
		}

		if(cudaMemcpy(GPUtemp, output, sizeof(unsigned char)*inW*inH, cudaMemcpyHostToDevice)!=cudaSuccess){ 
			printf ("cudaMemcpyHostToDevice Error!\n");	
			getchar();
			cudaFree(GPUtemp);
		}else{
			printf ("cudaMemcpyHostToDevice Success!\n");	
		}

		printf("dummy CUDA kernel call.. \n");

		GPUDummyKernel<<<inH,					//blocks=height of image
						inW>>>					//threads per block = width of image(max 1024)
						(GPUtemp,				//pointer to destination
						inW, inH);				//destination dimensions

		if(cudaMemcpy(output, GPUtemp, sizeof(unsigned char)*inW*inH, cudaMemcpyDeviceToHost)!=cudaSuccess){
			printf ("cudaMemcpyDeviceToHost Error!\n");	
			getchar();
			cudaFree(GPUtemp);
		}else{
			printf ("cudaMemcpyDeviceToHost Success!\n");	
		}

		cudaFree(GPUtemp);
	}
}
