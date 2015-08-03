/***********************************************************************************************************************
This file is part of "SIFT-GPU-Thesis" project.
 * Copyright (C) 2015 {Christos Papazachariou} <{christospapazachariou@gmail.com}>
 * University of Piraeus- MSc program: Advanced Information Systems ("Embedded System Technologies" branch).
 * Partial or otherwise use of this project and/or source code is permitted only for educational and academic purposes.
 * Partial or otherwise use of this project and/or source code for commercial or militay applications is prohibited.
***********************************************************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <conio.h>

#include "img_io.h"
#include "sift.h"

#include "cuda.h"
#include "cuda_runtime.h"
#include "gpusift.h"
#include "ioocv.h"
#include "MYtimer.h"
#include "GPUDummy.h"
#include "winfiles.h"
#include <windows.h>

int main(int argc, char* argv[]) {
	//COMMON VARIABLES
    int w, h;
    unsigned char* in_img;

	char ch;
	char* outfilename;
	char* infilename;
	char tempstring[300];
	const int testflag=0;		//flag for startup test
	const int customIOflag=1;	//IO mode selection flag
	int modeselect;				//GPU / CPU execution

		char** filelist;
		char* foldername="C:\\Users\\sleazy tek\\Documents\\Visual Studio 2010\\Projects\\SIFT-GPU-Thesis\\Debug\\HDLOGOS";
		char * filetype=".jpg";
		int fileindex=0;
		int filenumber=0;

		IplImage* ocvTemp;	//OpenCV compatibility test image
		unsigned char* gputest;	//cuda test image array
		int gpuX=255;	//CUDA test array dimensions
		int gpuY=255;	//CUDA test array dimensions

	IplImage* ocvInput;		//OpenCV IO usage

	IplImage* ocvResultCPU;
	IplImage* ocvResultGPU;

	unsigned char* ocvIn_img;	//OpenCV to project adaptor
	unsigned char* colorArray[3];
	int ocvInH;					//OpenCV input image height
	int ocvInW;					//OpenCV input image width
	HANDLE textcolor =GetStdHandle( STD_OUTPUT_HANDLE);

	displayflag=0;			//0=no show, 1 =show images
	saveflag=1;				//0=no save, 1=save images
	showmemoryflag=0;
	showtimeflag=0;
	
	setSeparator((char)219);	//set text separator format
		timeindex=1;				//global initialize
	timerstart=cvGetTickCount();	//set timing start point
	
	SetConsoleTextAttribute(textcolor,0x07 | FOREGROUND_INTENSITY);
	//PROJECT IMAGE IO SELECT
	if (customIOflag==0){	//DEFAULT PROJECT IO mode
		if (argc != 3) {
			fprintf(
				stderr, 
				"Usage: %s input-image output-image. Using defaults.\n",argv[0]);
				argv[1]="box.pgm";
				argv[2]="result.pgm";
		}//if (argc != 3)

		printf(" input image: %s \n output image: %s \n",argv[1], argv[2]);
		in_img = read_pgm(&w, &h, argv[1]);
		assert(in_img != NULL);
	}else if(customIOflag==1){	//OPENCV IO mode
		if (argc != 3) {
        fprintf(
            stderr, 
            "Usage: %s inputImage outputImage.\n  Using defaults...\n",argv[0]);
			//argv[1]="lena.jpg";
			argv[1]="NVIDIA_1080p.jpg";
			argv[2]="result.pgm";
		}//if (argc != 3)
		infilename=argv[1];
		outfilename=argv[2];
		printf(" Input image: %s\n Output Image: %s\n",infilename,outfilename);	
		ocvInput=cvLoadImage(infilename,CV_LOAD_IMAGE_COLOR);
			  if (!ocvInput){
					printf("OpenCV: Failed to open image!\n");
			  }else{
					ocvInH=ocvInput->height;		//get image dimensions
					ocvInW=ocvInput->width;
					h=ocvInH;						//pass dimensions to project variables
					w=ocvInW;	
					printf("Successfuly opened image.\n image height: %d pixels \n image width: %d pixels \n",ocvInH,ocvInW);
						cvShowImage("Original Image", ocvInput);
						cvWaitKey(SHOWDELAY);	//display original image for 2.5 sec
						cvDestroyWindow("Original Image");

					ocvIn_img=(unsigned char*)malloc(ocvInH*ocvInW*sizeof(unsigned char));

					colorArray[0]=(unsigned char*)malloc(ocvInH*ocvInW*sizeof(unsigned char));
					colorArray[1]=(unsigned char*)malloc(ocvInH*ocvInW*sizeof(unsigned char));
					colorArray[2]=(unsigned char*)malloc(ocvInH*ocvInW*sizeof(unsigned char));
					ocv2arrColor(ocvInput,colorArray,h,w);
					
					grayscale(colorArray,ocvIn_img,h,w);
					openShow2(ocvIn_img,h,w,"CPU Grayscale");
					openSave(ocvIn_img,h,w,"CPUgrayscale.jpg");

					GPUgrayscale(colorArray,ocvIn_img,h,w);
					openShow2(ocvIn_img,h,w,"GPU Grayscale");
					openSave(ocvIn_img,h,w,"GPUgrayscale.jpg");

					in_img=(unsigned char*)malloc(h*w*sizeof(unsigned char));	//allocate project variable
					memcpy(in_img,ocvIn_img,sizeof(unsigned char)*h*w);	//copy input array to project variable
							
			  }//else
	}//else if(customIOflag==1

	if (testflag==1){
		//OPENCV TEST COMPATIBILITY
		printf("Testing OpenCV compatibility...");
		ocvTemp=cvCreateImage(cvSize(w, h),8,1);
		arr2ocv(in_img,ocvTemp,h,w);
				cvShowImage("OCV temp", ocvTemp);
				cvWaitKey(SHOWDELAY);
				cvDestroyWindow("OCV temp");
				cvReleaseImage(&ocvTemp);
		//CUDA TEST COMPATIBILITY +OPENCV
				printf("Testing CUDA compatibility.\n");
				gputest=(unsigned char*)malloc(gpuX*gpuY*sizeof(unsigned char));
				dummyWrapper(gputest,gpuX,gpuY);
				ocvTemp=cvCreateImage(cvSize(gpuX,gpuY),8,1);
				arr2ocv(gputest,ocvTemp,gpuX,gpuY);
				cvShowImage("GPU temp", ocvTemp);
				cvWaitKey(SHOWDELAY);
				cvDestroyWindow("GPU temp");
				cvReleaseImage(&ocvTemp);	//cleanup OpenCV memory
				free(gputest);	//cleanup array memory
				printf("CUDA compatibility test complete.\n");
	}//	if (testflag==1)		

	SetConsoleTextAttribute(textcolor,0x03 | FOREGROUND_INTENSITY);
	printf("Please select mode:\n\n\t1=CPU\n\n\t2=GPU\n\n\t3=CPU+GPU and compare results\n\n\t4=multi-file GPU + GPU with comparisson\n\t\tWARNING:this will execute all functions\n\t\tfor every image file\n\t\tin the given folder\n\n");
	scanf("%d",&modeselect);
	
	switch (modeselect){
	case 1:
				SetConsoleTextAttribute(textcolor,0x06 | FOREGROUND_INTENSITY);
					logtime("CPU SIFT-start");
				printf("\n selected: CPU \n");
				sift(in_img, w, h, argv[2]);
					logtime("CPU SIFT-finish");
					showtimes();
					showtimes2();
					savetimes(infilename);
					savetimes2(infilename);

					if(customIOflag==1){
						ocvResultCPU=cvLoadImage(outfilename,CV_LOAD_IMAGE_COLOR);
						saveWrapper(ocvResultCPU,"CPU-SIFT-RESULT.JPG");
						if(remove(argv[2])!=0){
							printf("Error deleting temp file!\n");
						}else{
							printf("Temp file deleted successfully!\n");
						}
					}
					SetConsoleTextAttribute(textcolor,0x07 | FOREGROUND_INTENSITY);
					break;
	case 2:
				SetConsoleTextAttribute(textcolor,FOREGROUND_GREEN | FOREGROUND_INTENSITY);
					logtime("GPU SIFT-start");
				printf("\n selected: GPU \n");
				GPUsift(in_img, w, h, argv[2]);
					logtime("GPU SIFT-finish");
					showtimes();
					showtimes2();
					savetimes(infilename);
					savetimes2(infilename);

					if(customIOflag==1){
						ocvResultGPU=cvLoadImage(outfilename,CV_LOAD_IMAGE_COLOR);
						saveWrapper(ocvResultGPU,"GPU-SIFT-RESULT.JPG");
						if(remove(argv[2])!=0){
							printf("Error deleting temp file!\n");
						}else{
							printf("Temp file deleted successfully!\n");
						}
					}
					SetConsoleTextAttribute(textcolor,0x07 | FOREGROUND_INTENSITY);
					break;
	
	case 3:
				SetConsoleTextAttribute(textcolor,0x07 | FOREGROUND_INTENSITY);
				printf ("\n selected: GPU + GPU with comparisson\n");
				SetConsoleTextAttribute(textcolor,0x06 | FOREGROUND_INTENSITY);
				printf ("Starting CPU run of file %s.\n",argv[1]);
					logtime("CPU SIFT-start");
				sift(in_img, w, h, argv[2]);
					logtime("CPU SIFT-finish");
			
				ocvResultCPU=cvLoadImage(outfilename,CV_LOAD_IMAGE_COLOR);
				saveWrapper(ocvResultCPU,"CPU-SIFT-RESULT.JPG");
				if(remove(argv[2])!=0){
						printf("Error deleting temp file!\n");
					}else{
						printf("Temp file deleted successfully!\n");
					}
				SetConsoleTextAttribute(textcolor,FOREGROUND_GREEN | FOREGROUND_INTENSITY);
				printf ("Starting GPU run of file %s.\n",argv[1]);
					logtime("GPU SIFT-start");
				GPUsift(in_img, w, h, argv[2]);
					logtime("GPU SIFT-finish");
					showtimes();
					showtimes2();
					savetimes(infilename);
					savetimes2(infilename);
			
				ocvResultGPU=cvLoadImage(outfilename,CV_LOAD_IMAGE_COLOR);
				saveWrapper(ocvResultGPU,"GPU-SIFT-RESULT.JPG");
				if(remove(argv[2])!=0){
						printf("Error deleting temp file!\n");
					}else{
						printf("Temp file deleted successfully!\n");
					}
		
				SetConsoleTextAttribute(textcolor,0x07 | FOREGROUND_INTENSITY);
				ocvTemp=cvCreateImage(cvSize(w, h),8,3);
				cvSub(ocvResultCPU,ocvResultGPU,ocvTemp,0);
			
				cvShowImage("differences",ocvTemp);
				cvWaitKey(LONGDELAY);
				cvDestroyWindow("differences");
				saveWrapper(ocvTemp,"Result-comparisson.jpg");
				cvReleaseImage(&ocvResultCPU);
				cvReleaseImage(&ocvResultGPU);
				cvReleaseImage(&ocvTemp);
				break;

	case 4:
				showmemoryflag=0;
				showtimeflag=0;
				SetConsoleTextAttribute(textcolor,0x04 | FOREGROUND_INTENSITY);
				printf ("\n selected: multi-file GPU + GPU with comparisson\n");
				printf("\tcounting files...\n");
				filenumber=countfiles(foldername,filetype);
				printf("\tallocating memory for filelist...\n");
				filelist=(char**)malloc(filenumber*sizeof(char*));
				printf("\tallocating memory for filelist contents...\n");
				for (fileindex=0;fileindex<filenumber;fileindex++){
					filelist[fileindex]=(char*)malloc(MAXCHARNUMBER*sizeof(char));
				}
				printf("\tretriving filenames...\n");
				getfilelist(foldername,filelist,filetype);
				for (fileindex=0;fileindex<filenumber;fileindex++){
					printf("%03d\t%s\t",fileindex,filelist[fileindex]);
					if (fileindex%5==0){
						printf("\n");
					}
					printf("\n");
				}

				for (fileindex=0;fileindex<filenumber;fileindex++){
					timeindex=0;
					sprintf(tempstring,"%s\\%s",foldername,filelist[fileindex]);
					SetConsoleTextAttribute(textcolor,FOREGROUND_RED | FOREGROUND_INTENSITY);
					printf("opening: %s\n",tempstring);
					SetConsoleTextAttribute(textcolor,0x07 | FOREGROUND_INTENSITY);
					ocvInput=cvLoadImage(tempstring,CV_LOAD_IMAGE_COLOR);
					if (!ocvInput){
						printf("OpenCV: Failed to open image!\n");
					}else{
						ocvInH=ocvInput->height;		//get image dimensions
						ocvInW=ocvInput->width;
						h=ocvInH;						//pass dimensions to project variables
						w=ocvInW;	
						printf("Successfuly opened image.\n image height: %d pixels \n image width: %d pixels \n",ocvInH,ocvInW);
						cvShowImage("Original Image", ocvInput);
						cvWaitKey(SHOWDELAY);	//display original image for 2.5 sec
						cvDestroyWindow("Original Image");
						ocvIn_img=(unsigned char*)malloc(ocvInH*ocvInW*sizeof(unsigned char));

						colorArray[0]=(unsigned char*)malloc(ocvInH*ocvInW*sizeof(unsigned char));
						colorArray[1]=(unsigned char*)malloc(ocvInH*ocvInW*sizeof(unsigned char));
						colorArray[2]=(unsigned char*)malloc(ocvInH*ocvInW*sizeof(unsigned char));
						ocv2arrColor(ocvInput,colorArray,h,w);
					
						grayscale(colorArray,ocvIn_img,h,w);
						openShow2(ocvIn_img,h,w,"CPU Grayscale");
						openSave(ocvIn_img,h,w,"CPUgrayscale.jpg");

						GPUgrayscale(colorArray,ocvIn_img,h,w);
						openShow2(ocvIn_img,h,w,"GPU Grayscale");
						openSave(ocvIn_img,h,w,"GPUgrayscale.jpg");
					
						in_img=(unsigned char*)malloc(h*w*sizeof(unsigned char));	//allocate project variable
						memcpy(in_img,ocvIn_img,sizeof(unsigned char)*h*w);	//copy input array to project variable
					}

						printf ("Starting CPU run of file %s.\n",filelist[fileindex]);
							logtime("CPU SIFT-start");
						sift(in_img, w, h, argv[2]);
							logtime("CPU SIFT-finish");
			
						ocvResultCPU=cvLoadImage(outfilename,CV_LOAD_IMAGE_COLOR);
						sprintf(tempstring,"mode4/CPU-SIFT-RESULT_%s.jpg",filelist[fileindex]);
						saveWrapper(ocvResultCPU,tempstring);
						if(remove(argv[2])!=0){
								printf("Error deleting temp file!\n");
							}else{
								printf("Temp file deleted successfully!\n");
							}
						SetConsoleTextAttribute(textcolor,FOREGROUND_GREEN | FOREGROUND_INTENSITY);
						printf ("Starting GPU run of file %s.\n",filelist[fileindex]);
							logtime("GPU SIFT-start");
						GPUsift(in_img, w, h, argv[2]);
							logtime("GPU SIFT-finish");
						//showtimes();
						savetimes(filelist[fileindex]);
						savetimes2(filelist[fileindex]);
			
						ocvResultGPU=cvLoadImage(outfilename,CV_LOAD_IMAGE_COLOR);
						sprintf(tempstring,"mode4/GPU-SIFT-RESULT_%s.jpg",filelist[fileindex]);
						saveWrapper(ocvResultGPU,tempstring);
						if(remove(argv[2])!=0){
								printf("Error deleting temp file!\n");
							}else{
								printf("Temp file deleted successfully!\n");
							}
		
						SetConsoleTextAttribute(textcolor,0x07 | FOREGROUND_INTENSITY);
						ocvTemp=cvCreateImage(cvSize(w, h),8,3);
	
						cvReleaseImage(&ocvResultCPU);
						cvReleaseImage(&ocvResultGPU);
						cvReleaseImage(&ocvTemp);

				}
				free(filelist);
			
	default:
			//}else{
				SetConsoleTextAttribute(textcolor,0x07 | FOREGROUND_INTENSITY);
				 printf("Operations completed.\n");
			//}//if modeselect
				 break;

	}//switch

	printf("Press ESCape to exit.\n");
	while((ch = getch()) != 27);
	cvReleaseImage(&ocvInput);
	printf("OpenCV memory cleared\n");
	//cleanup project array memory before exit
	free(ocvIn_img);
	printf("ocvIn_img free\n");
	free(in_img);
	printf("Memory cleared. Exiting...\nThank You!");
	cvWaitKey(SHOWDELAY);
return 0;
}// main()
