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

extern "C" {
	void GPUbuildSS(unsigned char* octaves[MAX_O],		//ptr to input image
					float* scaleSpace [MAX_O][MAX_S],	//ptr to output image
					int O, int S,						//octave no, blur level no
					int* octavesW, int* octavesH,		//ptr to dims of output
					float sigmas[MAX_O][MAX_S]){		//input sigma for each scale

			//COMMON VARS
			int i=0;
			int j=0;
			int ii=0;

			int gR[MAX_O][MAX_S];
			int gW[MAX_O][MAX_S];	//gauss coefficient window width
			float* coef1d[MAX_O][MAX_S];
			float tmp;
			float norm = 0.0f;

			//CUDA VARS
			unsigned char* GPUoctaves[MAX_O];
			float* GPUscaleSpace[MAX_O][MAX_S];
			float* GPUcoef1d[MAX_O][MAX_S];		//TODO:CONSTANT MEMORY
			float* GPUintermediate[MAX_O][MAX_S];
			unsigned char* GPUsrcPadded[MAX_O][MAX_S];
			printf("GPUbuildSS call.\n");

			//CPU MALLOC + CALCULATE PADDINGS
			printf("  calculating paddings and allocating RAM...\n");
			logtime("GPUbuildSS-CPU padding+coef1d malloc loop start");
			for (i=0;i<O;i++){		//for each octave
				for (j=0;j<S;j++){	//for each scale
					// derive gR and gW
					if(sigmas[i][j]*4.0f > 1.0f){
						gR[i][j] = (int)(sigmas[i][j] * 4.0f);
					}else{
						gR[i][j] = 1;
					}
					gW[i][j] = gR[i][j] * 2 + 1;
					//malloc the 1d gauss coefficient
					coef1d[i][j]=(float*)malloc(gW[i][j]*sizeof(float));
				}
			}
			logtime("GPUbuildSS-CPU padding+coef1d malloc loop end");

				/****************************************
				* Compute Gaussian Coefficients
				***************************************/
			printf("  calculating Gaussian coefficients...\n");
			logtime("GPUbuildSS-CPU coef1d calc loop start");
			for (i=0;i<O;i++){		//for each octave
				for (j=0;j<S;j++){	//for each scale
					norm=0.0f;
					for(ii = 0; ii < gW[i][j]; ii++){
						tmp = (float)((float)ii - (float)gR[i][j]) / sigmas[i][j];
						coef1d[i][j][ii] = exp( -1.0f * tmp * tmp / 2.0f );
						norm = norm + coef1d[i][j][ii];
					}
					for(ii = 0; ii < gW[i][j]; ii++){
						coef1d[i][j][ii] = coef1d[i][j][ii] / norm;
					}
				}
			}
			logtime("GPUbuildSS-CPU coef1d calc loop end");

	//CUDA MALLOCS
			printf("  allocating GPU memory...\n");
			showFreeBytes();			
			logtime("GPUbuildSS-cudaMalloc loops start");
			for (i=0;i<O;i++){		//for each octave
				if(cudaMalloc(&GPUoctaves[i],sizeof(unsigned char)*octavesW[i]*octavesH[i])!=cudaSuccess){
					cudaError_t error = cudaGetLastError();
					printf ("cudaMalloc Error! GPUoctaves[%d]\tCUDA error: %s\n",i,cudaGetErrorString(error));	
					cudaFree(GPUoctaves[i]);
				}
			}

			showFreeBytes();
			for (i=0;i<O;i++){		//for each octave
				for (j=0;j<S;j++){	//for each scale
					if (cudaMalloc(&GPUscaleSpace[i][j],sizeof(float)*octavesH[i]*octavesW[i])!=cudaSuccess){
						cudaError_t error = cudaGetLastError();
						printf ("cudaMalloc Error! GPUscaleSpace[%d][%d]\tCUDA error: %s\n",i,j,cudaGetErrorString(error));	
						cudaFree(GPUscaleSpace[i][j]);
					}
				}
			}

			showFreeBytes();
			for (i=0;i<O;i++){		//for each octave
				for (j=0;j<S;j++){	//for each scale
					if (cudaMalloc(&GPUcoef1d[i][j],sizeof(float)*gW[i][j])!=cudaSuccess){
						cudaError_t error = cudaGetLastError();
						printf ("cudaMalloc Error! GPUcoef1d[%d][%d]\tCUDA error: %s\n",i,j,cudaGetErrorString(error));	
						cudaFree(GPUcoef1d[i][j]);
					}
				}
			}

			showFreeBytes();
			for (i=0;i<O;i++){		//for each octave
				for (j=0;j<S;j++){	//for each scale
					if (cudaMalloc(&GPUsrcPadded[i][j],sizeof(unsigned char)*(octavesH[i]+gR[i][j]*2)*(octavesW[i]+gR[i][j]*2))!=cudaSuccess){
						cudaError_t error = cudaGetLastError();
						printf ("cudaMalloc Error! GPUsrcPadded[%d][%d]\tCUDA error: %s\n",i,j,cudaGetErrorString(error));	
						cudaFree(GPUsrcPadded[i][j]);
					}
				}
			}

			showFreeBytes();
			for (i=0;i<O;i++){		//for each octave
				for (j=0;j<S;j++){	//for each scale
					if (cudaMalloc(&GPUintermediate[i][j],sizeof(float)*(octavesH[i]+gR[i][j]*2)*(octavesW[i]+gR[i][j]*2))!=cudaSuccess){
						cudaError_t error = cudaGetLastError();
						printf ("cudaMalloc Error! GPUintermediate[%d][%d]\tCUDA error: %s\n",i,j,cudaGetErrorString(error));	
						cudaFree(GPUintermediate[i][j]);
					}
				}
			}
			logtime("GPUbuildSS-cudaMalloc loops end");
			showFreeBytes();

	//CUDA MEMCPYS TO DEV
			printf("  copying data to GPU memory...\n");
			logtime("GPUbuildSS-cudaMemcpyHostToDevice loops start");
			for (i=0;i<O;i++){
					if(cudaMemcpy(GPUoctaves[i],octaves[i],sizeof(unsigned char)*octavesW[i]*octavesH[i],cudaMemcpyHostToDevice)!=cudaSuccess){
						cudaError_t error = cudaGetLastError();
						printf ("cudaMemcpyHostToDevice Error! GPUoctaves[%d]\tCUDA error: %s\n",i,cudaGetErrorString(error));	
						cudaFree(GPUoctaves[i]);
					}
			}

			for (i=0;i<O;i++){
				for (j=0;j<S;j++){
					if(cudaMemcpy(GPUcoef1d[i][j],coef1d[i][j],sizeof(float)*gW[i][j],cudaMemcpyHostToDevice)!=cudaSuccess){
						cudaError_t error = cudaGetLastError();
						printf ("cudaMemcpyHostToDevice Error! GPUcoef1d[%d][%d]\tCUDA error: %s\n",i,j,cudaGetErrorString(error));	
						cudaFree(GPUoctaves[i]);
					}
				}
			}
			logtime("GPUbuildSS-cudaMemcpyHostToDevice loops end");

	//CUDA KERNEL CALLS
			printf("  calling padding GPU kernel...\n");
			logtime("GPUbuildSS-Padding Kernel loop start");
			for (i=0;i<O;i++){
				for (j=0;j<S;j++){

						const dim3 block(32,32);	//reasonable block size (1024 threads/block) 
						const dim3 grid((octavesW[i]+gR[i][j]*2+block.x-1)/block.x,(octavesH[i]+gR[i][j]*2+block.x-1)/block.y);	//make sure at least 1x1 kernel

					GPUsrcPadding<<<grid,			
						block>>>					
						(GPUoctaves[i],					//input image (uchar*)
						GPUsrcPadded[i][j],				//output image (uchar*)
						GPUintermediate[i][j],			//output image (float*)
						gR[i][j],						//gaussian coef radius
						octavesH[i],octavesW[i]);		//input image dimensions
				}
			}
			logtime("GPUbuildSS-Padding Kernel loop end");

			printf("  calling GPUGaussX kernel...\n");
			logtime("GPUbuildSS-GaussX Kernel loop start");
			for (i=0;i<O;i++){
				for (j=0;j<S;j++){

						const dim3 block(32,32);	//reasonable block size (1024 threads/block) 
						const dim3 grid((octavesW[i]+gR[i][j]*2+block.x-1)/block.x,(octavesH[i]+gR[i][j]*2+block.x-1)/block.y);	//make sure at least 1x1 kernel

					GPUGaussX<<<grid,				
						block>>>					
						(GPUsrcPadded[i][j],			//input image (uchar*)
						GPUintermediate[i][j],			//output image (float*)
						GPUcoef1d[i][j],				//input gaussian coef 1D array
						gR[i][j],						//gaussian coef radius
						octavesH[i],octavesW[i]);		//input image dimensions
				}
			}
			logtime("GPUbuildSS-GaussX Kernel loop end");

			printf("  calling GPUGaussY kernel...\n");
			logtime("GPUbuildSS-GaussY Kernel loop start");
			for (i=0;i<O;i++){
				for (j=0;j<S;j++){

						const dim3 block(32,32);	//reasonable block size (1024 threads/block) 
						const dim3 grid((octavesW[i]+gR[i][j]*2+block.x-1)/block.x,(octavesH[i]+gR[i][j]*2+block.x-1)/block.y);	//make sure at least 1x1 kernel

					GPUGaussY<<<grid,			
						block>>>				
						(GPUintermediate[i][j],		//input image (float*)
						GPUscaleSpace[i][j],		//output image(float*)
						GPUcoef1d[i][j],			//input gaussian coef 1D array
						gR[i][j],					//gaussian coef radius
						octavesH[i],octavesW[i]);	//input image dimensions
				}
			}
			logtime("GPUbuildSS-GaussY Kernel loop end");

	//CUDA MEMCPYS TO HOST
			printf("  copying data from GPU memory to RAM...\n");
			showFreeBytes();
			logtime("GPUbuildSS-cudaMemcpyDeviceToHost loop start");
			for (i=0;i<O;i++){
				for (j=0;j<S;j++){
					if(cudaMemcpy(scaleSpace[i][j],GPUscaleSpace[i][j],sizeof(float)*octavesW[i]*octavesH[i],cudaMemcpyDeviceToHost)!=cudaSuccess){
						cudaError_t error = cudaGetLastError();
						printf ("cudaMemcpyDeviceToHost Error! scaleSpace[%d][%d]\tCUDA error: %s\n",i,j,cudaGetErrorString(error));	
						cudaFree(GPUscaleSpace[i][j]);
					}
				}
			}
			logtime("GPUbuildSS-cudaMemcpyDeviceToHost loop end");
			showFreeBytes();

	//CUDAFREES
			printf("  clearing GPU memory...\n");
			cudaFree(GPUscaleSpace);
			cudaFree(GPUoctaves);
			cudaFree(GPUcoef1d);
			cudaFree(GPUsrcPadded);
			cudaFree(GPUintermediate);
			showFreeBytes();

	//CPU MEMORY CLEANUP
			printf("  clearing RAM...\n");
			//free(coef1d); //gia kapoio logo crasharei edw
			printf("GPUbuildSS complete!\n");
	}
}

extern "C" {
	__global__ void GPUsrcPadding(unsigned char* srcIn,				//input image
									unsigned char* srcPaddedOut,	//output image
									float* intermediate,			//output image
									int gR,							//input gaussian radius
									int inH, int inW){				//input image dims

								const int x=blockIdx.x*blockDim.x+threadIdx.x;
								const int y=blockIdx.y*blockDim.y+threadIdx.y;

									if ((x<inW+gR) && (y<inH+gR)){		//only valid threads
										if( x < gR || y > inH+gR-1 
											|| y < gR || x > inW+gR-1 ){
											// padding srcPadded
											srcPaddedOut[y*(inW+gR*2)+x] = 0;
											// padding intermediate
											intermediate[y*(inW+gR*2)+x] = 0.0f;
										}else{
											// copy the src to srcPadded
											srcPaddedOut[y*(inW+gR*2)+x] = 
												srcIn[(y-gR)*inW+(x-gR)];
										}
									}
	}
}

extern "C" {
	__global__ void GPUGaussX(unsigned char* input,		//input image (sourcePadded)
								float* output,			//output image (intermediate)
								float* coef1d,			//input gaussian coefficient 1d
								int gR,					//input gaussian radius
								int inH, int inW){		//input image dims

						const int x=blockIdx.x*blockDim.x+threadIdx.x;
						const int y=blockIdx.y*blockDim.y+threadIdx.y;

										float accu = 0.0f;
										int ii=0;
										if (x>=gR && y>=gR && y<inH+gR && x<inW+gR){	//for loop starts at gR for i and j...
											for(ii = (-1 * gR); ii < (gR + 1); ii++){
												// convolute with gaussian window
												// note: srcPadded is unsigned char
												accu += 
													coef1d[ii + gR] * 
													(float)input[y*(inW+gR*2)+x+ii];
											}
										output[(y)*(inW+gR*2)+(x)] = accu;
										}
	}
}

extern "C" {
	__global__ void GPUGaussY(float* input,				//input image (intermediate)
								float* output,			//output image (blurred)
								float* coef1d,			//input gaussian coefficient 1d
								int gR,					//input gaussian radius
								int inH, int inW){		//input image dims
										float accu = 0.0f;
										int ii=0;

						const int x=blockIdx.x*blockDim.x+threadIdx.x;
						const int y=blockIdx.y*blockDim.y+threadIdx.y;

										if (x>=gR && y>=gR && y<inH+gR && x<inW+gR){	//for loop starts at gR for i and j...
											for(ii = (-1 * gR); ii < (gR + 1); ii++){
												// convolute with gaussian window
												accu += 
													coef1d[ii + gR] * 
													(float)
													input[(y+ii)*(inW+gR*2)+x];
											}
											output[(y-gR)*inW+(x-gR)] = accu;	
										}
	}
}
