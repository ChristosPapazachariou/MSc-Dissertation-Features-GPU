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

extern "C"{ 
	__global__ void GPUdownsample(unsigned char* target,
                int targetW, int targetH,
                unsigned char* src,
                int srcW, int srcH){
				
				//set x and y from block and thread ids
				const int x=blockIdx.x*blockDim.x+threadIdx.x;
				const int y=blockIdx.y*blockDim.y+threadIdx.y;

				if ((x<srcW) && (y<srcH)){
					target[y*targetW+x] = (unsigned char)
                ( (int)
                  ((int)src[y*2*srcW+x*2] +
                   (int)src[y*2*srcW+x*2+1] +
                   (int)src[(y*2+1)*srcW+x*2] +
                   (int)src[(y*2+1)*srcW+x*2+1]) 
                  / 4 );
				}

	}//GPUdownsample
}//extern "C"

extern "C" {
	void GPUbuildOB(unsigned char* img,			//pointer to input image (source)
             unsigned char* octaves[MAX_O],		//array of pointers to output images (destinationS)
             int O, int S,						//number of octaves and blur levels
             int* octavesW, int* octavesH){		//pointer to array of dimensions of output images
  
// common variables
    int i;
	unsigned char *GPUimg;				//declare pointer to gpu source to later malloc and copy
	unsigned char* GPUoctaves[MAX_O];	//declare pointer to array of pointers-to-(downsampled) images in gpu memory

//CUDAMALLOCS
	printf("GPUbuildOB call. \n");
	printf("  starting gpu  memory allocation... \n");
	showFreeBytes();
	logtime("GPUbuildOB-cudaMalloc");
	if(cudaMalloc(&GPUimg,sizeof(unsigned char)*octavesW[0]*octavesH[0])!=cudaSuccess){		//cuda malloc for source image
		cudaError_t error = cudaGetLastError();
		printf ("cudaMalloc Error! GPUimg\tCUDA error: %s\n",cudaGetErrorString(error));	
		cudaFree(GPUimg);
	}
	showFreeBytes();
	logtime("GPUbuildOB-cudaMalloc loop start");
	
	for(i=0;i<O;i++){	//for each octave
			if(cudaMalloc(&GPUoctaves[i],sizeof(unsigned char)*octavesW[i]*octavesH[i])!=cudaSuccess){	//allocate array of pointers in gpu memory size of the pixels of current octave*size of pixel
				cudaError_t error = cudaGetLastError();
				printf ("cudaMalloc Error! GPUoctaves[%d]\tCUDA error: %s\n",i,cudaGetErrorString(error));	
				cudaFree(GPUoctaves[i]);
			}
		}//for
	
//CUDAMEMCPYS
	logtime("GPUbuildOB-cudaMalloc loop end");
	showFreeBytes();
	printf("  starting gpu  memory copy... \n");
	logtime("GPUbuildOB-cudaMemcpyHostToDevice start");
	if(cudaMemcpy(GPUimg, img, sizeof(unsigned char)*octavesW[0]*octavesH[0], cudaMemcpyHostToDevice)!=cudaSuccess){	//cuda copy the source image to gpu 
			cudaError_t error = cudaGetLastError();
			printf ("cudaMemcpyHostToDevice Error! GPUimg\tCUDA error: %s\n",cudaGetErrorString(error));	
			cudaFree(GPUimg);
		}

	logtime("GPUbuildOB-cudaMemcpyHostToDevice end");

	if(cudaMemcpy(GPUoctaves[0], GPUimg, sizeof(unsigned char)*octavesW[0]*octavesH[0], cudaMemcpyDeviceToDevice)!=cudaSuccess){	//cuda copy the source image to octaves 
		cudaError_t error = cudaGetLastError();
		printf ("cudaMemcpyDeviceToDevice Error! GPUimg\tCUDA error: %s\n",cudaGetErrorString(error));
		cudaFree(GPUimg);
		cudaFree(GPUoctaves[0]);
	}
	showFreeBytes();

//KERNEL CALLS
	printf("  calling downsample kernel... \n");

	logtime("GPU downsample loop -start");

	for(i=0;i<O;i++){									//for each octave

			const dim3 block(32,32);	//reasonable block size (1024 threads/block) 
			const dim3 grid((octavesW[i+1]+block.x-1)/block.x,(octavesH[i+1]+block.x-1)/block.y);	//make sure at least 1x1 kernel
			
		GPUdownsample<<<grid,							
						block>>>						
						(GPUoctaves[i+1],				//pointer to destination pointer to current octave image (*uchar)
						octavesW[i+1], octavesH[i+1],	//pointer to destination dimensions
						GPUoctaves[i],					//pointer to source pointer of previous octave image (*uchar)
						octavesW[i], octavesH[i]);		//pointers to source dimensions
		}//for

	logtime("GPU downsample loop -end");

//copy downsampled octaves to host
	printf("  downloading from gpu... \n");
	showFreeBytes();
	logtime("GPUbuildOB-cudaMemcpyDeviceToHost loop start");
	for(i=0;i<O;i++){
				//printf("copying octave %d \n",i);
			if(cudaMemcpy(octaves[i], GPUoctaves[i], sizeof(unsigned char)*octavesW[i]*octavesH[i], cudaMemcpyDeviceToHost)!=cudaSuccess){	//cuda copy to RAM
				cudaError_t error = cudaGetLastError();
				printf ("cudaMemcpyDeviceToHost Error! GPUoctaves[%d]\tCUDA error: %s\n",i,cudaGetErrorString(error));	
				cudaFree(GPUoctaves[i]);
			}else{
				//printf ("cudaMemcpyDeviceToHost Success!\n");	
			}		
		}
	logtime("GPUbuildOB-cudaMemcpyDeviceToHost loop end");
	printf("  downloading from gpu complete. \n");
	showFreeBytes();
//release memory
	cudaFree(GPUimg);
	cudaFree(GPUoctaves);
	printf("  GPU memory released. \n");
	showFreeBytes();
	printf("GPUbuildOB complete.\n");
	
}//GPUbuildOB
}//extern "C"
