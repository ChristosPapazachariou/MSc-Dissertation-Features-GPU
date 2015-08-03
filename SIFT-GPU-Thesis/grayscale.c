/***********************************************************************************************************************
This file is part of "SIFT-GPU-Thesis" project.
 * Copyright (C) 2015 {Christos Papazachariou} <{christospapazachariou@gmail.com}>
 * University of Piraeus- MSc program: Advanced Information Systems ("Embedded System Technologies" branch).
 * Partial or otherwise use of this project and/or source code is permitted only for educational and academic purposes.
 * Partial or otherwise use of this project and/or source code for commercial or militay applications is prohibited.
***********************************************************************************************************************/

#include "sift.h"
#include "MYtimer.h"

void grayscale(unsigned char* input[3], unsigned char* output, int inH, int inW){
	int x=0;
	int y=0;
	float redFactor=0.2989;
	float greenFactor=0.5870;
	float blueFactor=0.1140;
	
	printf("Grayscale conversion call...\n");
	logtime("CPU Grayscale loop-start");
	
	for (y=0;y<inH;y++){	
		for (x=0;x<inW;x++){
		
			output[y*inW+x]=
				input[0][y*inW+x]*redFactor+
				input[1][y*inW+x]*greenFactor+
				input[2][y*inW+x]*blueFactor;
		}
	}
	
	logtime("CPU Grayscale loop-end");
	printf("Grayscale conversion ended.\n");

}