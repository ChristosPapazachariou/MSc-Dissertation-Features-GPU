/***********************************************************************************************************************
This file is part of "SIFT-GPU-Thesis" project.
 * Copyright (C) 2015 {Christos Papazachariou} <{christospapazachariou@gmail.com}>
 * University of Piraeus- MSc program: Advanced Information Systems ("Embedded System Technologies" branch).
 * Partial or otherwise use of this project and/or source code is permitted only for educational and academic purposes.
 * Partial or otherwise use of this project and/or source code for commercial or militay applications is prohibited.
***********************************************************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "sift.h"

#include "MYtimer.h"
#include "ioocv.h"

void ocv2arrColor(IplImage* input, unsigned char* output[3],int inH, int inW){
	int x=0;
	int y=0;
	int i=0;

	for (y=0;y<inH;y++){
		for (x=0;x<inW;x++){
		
			output[2][i]=CV_IMAGE_ELEM(input, unsigned char,y,3*x);
			output[1][i]=CV_IMAGE_ELEM(input, unsigned char,y,3*x+1);
			output[0][i]=CV_IMAGE_ELEM(input, unsigned char,y,3*x+2);
			i++;
		}
	}
}

void arr2ocv(unsigned char* input, IplImage* output, int inH, int inW){
	int x=0,y=0,i=0;
	//printf("arr2ocv call...");
	
	for (y=0;y<inH;y++){	
		for (x=0;x<inW;x++){
				CV_IMAGE_ELEM(output, unsigned char, y, x) = input[i];
				i++;
		}
	}
}

void ocv2arr(IplImage* input, unsigned char* output, int inH, int inW){
	int x=0,y=0,i=0;
	//printf("ocv2arr call...");
	
	for (y=0;y<inH;y++){	
		for (x=0;x<inW;x++){
				output[i]=CV_IMAGE_ELEM(input, unsigned char, y, x);
				i++;
		}
	}
}

void grayWrapper(IplImage* input,IplImage* output){
	cvCvtColor(input, output, CV_BGR2GRAY);
	//printf("grayscale image created.\n");
}

void saveWrapper(IplImage* input,char* filename){
	if (saveflag!=0){
		cvSaveImage(filename,input,0);	//0=no compression
		//printf("processed image saved as %s.\n",filename);
	}
}

void openWrapper(IplImage* openedImage, char*filename){
	openedImage=cvLoadImage(filename,CV_LOAD_IMAGE_COLOR);
	if(openedImage == NULL){
		printf("Could not locate image.");
		}
}

void openShow(unsigned char* input, int inH, int inW, char* debuglabel){
	IplImage* ocvTemp;
	if (displayflag!=0){
		ocvTemp=cvCreateImage(cvSize(inW, inH),8,1);
				//printf("openShow call: %s ",debuglabel);
		arr2ocv(input,ocvTemp,inH,inW);
				//printf("array dimensions: h=%d w=%d ...",inH,inW);
		cvShowImage(debuglabel, ocvTemp);
				//printf("Displaying image... ");
		cvWaitKey(SHOWDELAY);					//display image with delay
		cvDestroyWindow(debuglabel);
		cvReleaseImage(&ocvTemp);
				//printf("Complete!\n");
	}
}

	void openShow2(unsigned char* input, int inH, int inW, char* debuglabel){
		IplImage* ocvTemp;
		if (displayflag!=0){
			ocvTemp=cvCreateImage(cvSize(inW, inH),8,1);
					//printf("openShow call: %s ",debuglabel);
			arr2ocv(input,ocvTemp,inH,inW);
					//printf("array dimensions: h=%d w=%d ...",inH,inW);
			cvShowImage(debuglabel, ocvTemp);
					//printf("Displaying image... ");
			cvWaitKey(LONGDELAY);					//display image with delay
			cvDestroyWindow(debuglabel);
			cvReleaseImage(&ocvTemp);
					//printf("Complete!\n");
		}
	}

void openSave(unsigned char* input, int inH, int inW, char* filename){
	IplImage* ocvSaveTemp;
	if (saveflag!=0){
		ocvSaveTemp=cvCreateImage(cvSize(inW,inH),8,1);
			//printf ("openSave call: %s ...",filename);
		arr2ocv(input,ocvSaveTemp,inH,inW);
			saveWrapper(ocvSaveTemp,filename);
		cvReleaseImage(&ocvSaveTemp);
			//printf ("Complete!\n");
	}
}

void openSaveFloat(float*input, char*filename,int inH, int inW){
	int y=0;
	int x=0;
	unsigned char* temparray;
	char tempname[200];
	if (saveflag!=0){
		temparray=(unsigned char*)malloc(inW*inH*sizeof(unsigned char));
		//printf(" float cast ");
		for (y=0;y<inH;y++){
			for (x=0;x<inW;x++){
				//printf("%02.2f\t",target[y*w+x]);
				temparray[y*inW+x]=(unsigned char)input[y*inW+x];
				}
			//printf("\n");
		}
		sprintf(tempname,filename);
		//sprintf(tempname,"%s %dx%d",filename,inH,inW);
		openSave(temparray,inH,inW,tempname);
		free(temparray);
	}
}

void openShowFloat(float*input, char*windowname,int inH, int inW){
	int y=0;
	int x=0;
	unsigned char* temparray;
	char tempname[200];
	if (displayflag!=0){
		temparray=(unsigned char*)malloc(inW*inH*sizeof(unsigned char));
		//printf(" float cast ");
		for (y=0;y<inH;y++){
			for (x=0;x<inW;x++){
				//printf("%02.2f\t",target[y*w+x]);
				temparray[y*inW+x]=(unsigned char)input[y*inW+x];
				}
			//printf("\n");
		}
		sprintf(tempname,"%s %dx%d",windowname,inH,inW);
		openShow(temparray,inH,inW,tempname);
		free(temparray);
	}
}
