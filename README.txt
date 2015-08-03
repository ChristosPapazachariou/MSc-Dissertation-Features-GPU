******************************************************************************
The original C implementation was created by Z. Ye and D. She 
(Eindhoven Polytechnic University 2011) and can be found here:
https://sites.google.com/site/5kk73gpu2011/assignments/sift/sift-c.zip?attredirects=0
The copyright specifications of that project can be read 
at the included README file
******************************************************************************

******************************************************************************
This enhanced project features several additions by:
Christos Papazachariou 

 * Copyright (C) 2015 {Christos Papazachariou} <{christospapazachariou@gmail.com}>
 * University of Piraeus- MSc program (2015):
 * Advanced Information Systems ("Embedded System Technologies" branch).
 * Partial or otherwise use of this project and/or source code is permitted only for educational and academic purposes.
 * Partial or otherwise use of this project and/or source code for commercial or militay applications is prohibited.
******************************************************************************

The overall structure of the project is:

Main function:
	Performs execution configurations for mode selection (CPU/GPU/both/both-multifile, file io mode (original/OpenCV), timing setup (original/OpenCV)
	Calls sift.c
	Calls GPUsift.c
	The grayscale conversion was not added to neither sift.c nor sift.cu because the original project did not feature it, as it only imported greyscale images, therefore on startup of OpenCV io mode, a greyscale conversion is performed for both CPU and GPU to measure and compare its performance. This can be ported inside the main SIFT calls


sift.c function:
	This calls the steps of the algorithm in CPU version (.c files). 
	Also handles debug messages and image display for verification of every step
	The functions are declared in sift.h


GPUsift.c function:
	this calls the steps of the algorithm in the GPU version (CUDA files). For any step that was not parallelized, it calls the serial version instead.
	Also handles debug messages and image display for verification of every step
	The functions are declared in GPUsift.h

The original timer.c was ignored as it was too complicated to use massively
A mytimer.c was created to handle an array of timelog structs that are defined in the MYtimer.h file. a call to the logtime function will store a timing measurement into the array and the array can be displayed and saved via the showtimes and savetimes functions
The 2 versions of the logtime, showtimes and savetimes functions are identical but where created to handle later measurements and calculations during the development of the project. The later ones can be ignored or replaced by the earlier ones if a uniform timetable is preferable.
the file mytimer.c also includes a version of openShow function (openShow3) that was moved here as this set of functions and this header are able to be safely called within CUDA functions. It only serves as a display/debug tool for viewing intermediate images from CUDA files.

ioocv.c  ioocv.h:
The original project did not feature a image display api, so a set of wrappers was created that call OpenCV v2.4.9 functions in order to perform view/debug of images during execution, and also manage saving and opening images in any format (jpg instead of just pgm that the original had).

greyscale.cu:
This file apart from the function kernels also contains a function that retrieves info about GPU memory usage. it was placed here as this was the shortest CUDA file to simplify the file structure of the project.

GPUdummy.cu   GPUdummy.h:
Those files were created to test the compatibility of the project with CUDA and configure the multi-compiler, multi-linker environment of the project. They can be ignored and their calls in main function can be removed if desired.

winfiles.c  winfiles.h:
Those files contain functions that were written according to Microsoft tutorials on win32 api and feature folder management and file management functions. They are used to retrieve the jpg contents of the project folders and store them to a list, in order to be used in mode4 (multi-file: CPU&GPU execution-comparison)

The steps of the algorithm are performed as in the original project with the following correlation to the CUDA versions:

buildOB.c	GPUbuildOB.cu
buildSS.c	GPUbuildSS.cu
DoG.c		GPUDoG.cu
extreme.c	GPUextreme.cu*

The GPUextreme* file contains 3 of the steps of the algorithm:
	extrema detection: succesfully parallelized
	edge elimination: parallelizable but the implemented version has unresolved bug(s) - Bypassed by a serial version that functions correctly
	keypoint registration: not parallelizable due to linked lists, instead implemented with stream compaction-like method	

the Visual Studio folders include the required folder structure for the operations of main on all modes, along with some sample pictures
a copy of all the required .dll files needed is included but it is recommended to setup the framework and libraries independently and replace them with the ones created on the project computer (OpenCV-CUDA dlls)
the Visual Studio folders also include 2 template property page files for debug and release mode that were created for this project
the CUDA dlls are those included in CUDA version 6.5 for a cc2.0 device (GTS450)
the OpenCV dlls were built from the OpenCV library source files using CMake and Visual studio 2010 ultimate.

For any questions on how to use or expand this project, contact me at christospapazachariou@gmail.com and I will try to answer promptly if I am available.



