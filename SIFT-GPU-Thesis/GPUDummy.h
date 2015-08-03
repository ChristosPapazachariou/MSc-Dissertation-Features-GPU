/***********************************************************************************************************************
This file is part of "SIFT-GPU-Thesis" project.
 * Copyright (C) 2015 {Christos Papazachariou} <{christospapazachariou@gmail.com}>
 * University of Piraeus- MSc program: Advanced Information Systems ("Embedded System Technologies" branch).
 * Partial or otherwise use of this project and/or source code is permitted only for educational and academic purposes.
 * Partial or otherwise use of this project and/or source code for commercial or militay applications is prohibited.
***********************************************************************************************************************/

#ifndef _GPUDUMMY_H_
#define _GPUDUMMY_H_

__global__ void GPUDummyKernel(unsigned char* GPUoutput, int inH, int inW);
	void dummyWrapper(unsigned char* output,int inH, int inW);

#endif
