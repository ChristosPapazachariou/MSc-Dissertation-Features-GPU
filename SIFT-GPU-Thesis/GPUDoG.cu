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
/**********************************************
 * Difference of Gaussians (DoG)
 * Only diff neighboring scales in same octave
 * Do not diff across two octaves
  * Parameters
 * dog: output, Ox(S-1) images in floating point
 * scaleSpace: input, OxS images in floating pt
 * O: number of octaves
 * S: scales per octave
 * octavesW: output, width of each octave
 * octavesH: output, height of each octave
 **********************************************/
extern "C"{
	void GPUDoG(float *dog[MAX_O][MAX_S-1], 
			 float *scaleSpace[MAX_O][MAX_S], 
			 int O, int S,
			 int* octavesW, int* octavesH){
		// common variables
		int i, j;

		//CUDA VARS
		float* GPUdog[MAX_O][MAX_S-1];
		float* GPUscaleSpace[MAX_O][MAX_S];
		printf("GPUDoG call\n");
	
		//CUDA MALLOCS
		showFreeBytes();
		logtime("GPUDoG-cudaMalloc GPUdog loop start");
		for (i=0; i<O; i++){
			for (j=0; j<S-1; j++){
				if(cudaMalloc(&GPUdog[i][j],sizeof(float)*octavesW[i]*octavesH[i])!=cudaSuccess){
						cudaError_t error = cudaGetLastError();
						printf ("cudaMalloc Error! GPUdog[%d][%d]\tCUDA error: %s\n",i,j,cudaGetErrorString(error));	
						cudaFree(GPUdog);
				}
			}
		}
		showFreeBytes();
		logtime("GPUDoG-cudaMalloc GPUscaleSPace loop start");
		for (i=0; i<O; i++){
			for (j=0; j<S; j++){
				if(cudaMalloc(&GPUscaleSpace[i][j],sizeof(float)*octavesW[i]*octavesH[i])!=cudaSuccess){
						cudaError_t error = cudaGetLastError();
						printf ("cudaMalloc Error! GPUscaleSpace[%d][%d]\tCUDA error: %s\n",i,j,cudaGetErrorString(error));
						cudaFree(GPUscaleSpace);
				}
			}
		}

		showFreeBytes();
		//CUDA MEMCPY TO DEV
		logtime("GPUDoG-cudaMemcpyHostToDevice loop start");
		for (i=0; i<O; i++){
			for (j=0; j<S; j++){
				if(cudaMemcpy(GPUscaleSpace[i][j],scaleSpace[i][j],sizeof(float)*octavesW[i]*octavesH[i],cudaMemcpyHostToDevice)!=cudaSuccess){
						cudaError_t error = cudaGetLastError();
						printf ("cudaMemcpyHostToDevice Error! GPUscaleSpace[%d][%d]\tCUDA error: %s\n",i,j,cudaGetErrorString(error));
						cudaFree(GPUscaleSpace);
				}
			}
		}
		logtime("GPUDoG-cudaMemcpyHostToDevice loop end");
		logtime("GPUDoG-DoG kernel loop start");

		//CUDA KERNEL CALLS
			for(i = 0; i < O; i++){						//for each octave
				for(j = 0; j < S-1; j++){				//for each blur scale
				
					const dim3 block(32,32);	//reasonable block size (1024 threads/block) 
					const dim3 grid((octavesW[i]+block.x-1)/block.x,(octavesH[i]+block.x-1)/block.y);	//make sure at least 1x1 kernel

					GPUdogKernel<<<grid,				
								block>>>				
								(GPUscaleSpace[i][j+1],		//input blurred image upper scale
								GPUscaleSpace[i][j],		//input blurred image lower scale
								GPUdog[i][j],				//output DoG image	
								octavesW[i],octavesH[i]);	//input images dims
				}
			}
			logtime("GPUDoG-DoG kernel loop end");

		//CUDA MEMCPY TO HOST
			showFreeBytes();
			logtime("GPUDoG- DoG cudaMemcpyDeviceToHost loop start");
			for (i=0; i<O; i++){
				for (j=0; j<S-1; j++){
					if(cudaMemcpy(dog[i][j], GPUdog[i][j],sizeof(float)*octavesW[i]*octavesH[i],cudaMemcpyDeviceToHost)!=cudaSuccess){
							cudaError_t error = cudaGetLastError();
							printf ("cudaMemcpyDeviceToHost Error! dog[%d][%d]\tCUDA error: %s\n",i,j,cudaGetErrorString(error));
							cudaFree(GPUdog);
					}
				}
			}
			logtime("GPUDoG- DoG cudaMemcpyDeviceToHost loop end");

			showFreeBytes();
				//CUDA FREE
			cudaFree(GPUdog);
			cudaFree(GPUscaleSpace);
		printf("GPUDoG complete!\n");
		showFreeBytes();
	}
}//extern "C"

extern "C"{
	__global__ void GPUdogKernel(float* SSU,	//input blurred image upper scale
					float* SSL,					//input blurred image lower scale
					float* dogout,				//output DoG image	
					int inW, int inH){			//input images dims
		
				const int x=blockIdx.x*blockDim.x+threadIdx.x;
				const int y=blockIdx.y*blockDim.y+threadIdx.y;

				if ((x<inW) && (y<inH)){	
					dogout[y*inW+x]=SSU[y*inW+x]-SSL[y*inW+x];	
				}
	}
}//extern "C"

