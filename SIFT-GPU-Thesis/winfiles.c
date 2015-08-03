/***********************************************************************************************************************
This file is part of "SIFT-GPU-Thesis" project.
 * Copyright (C) 2015 {Christos Papazachariou} <{christospapazachariou@gmail.com}>
 * University of Piraeus- MSc program: Advanced Information Systems ("Embedded System Technologies" branch).
 * Partial or otherwise use of this project and/or source code is permitted only for educational and academic purposes.
 * Partial or otherwise use of this project and/or source code for commercial or militay applications is prohibited.
***********************************************************************************************************************/

#include "winfiles.h"
#include <windows.h>
#include <tchar.h> 
#include <stdio.h>
#include <strsafe.h>
#pragma comment(lib, "User32.lib")

int countfiles(char* infoldername,	//input folder
		char* filetype){	//input filetype to count
	int filesfound=0;
	//char*infoldername=(STRSAFE_LPCWSTR)infoldernameC;;

	WIN32_FIND_DATA ffd;
   	TCHAR szDir[MAX_PATH];
	size_t length_of_arg;
	HANDLE hFind = INVALID_HANDLE_VALUE;
	DWORD dwError=0;

	// Check that the input path plus 3 is not longer than MAX_PATH.
	// Three characters are for the "\*" plus NULL appended below.
	StringCchLength(infoldername, MAX_PATH, &length_of_arg);
	if (length_of_arg > (MAX_PATH - 3)){
		_tprintf(TEXT("\nDirectory path is too long.\n"));
		exit(-1);
	}
	_tprintf(TEXT("Target directory is %s\n"), infoldername);
	// Prepare string for use with FindFile functions.  First, copy the
	// string to a buffer, then append '\*' to the directory name.
	StringCchCopy(szDir, MAX_PATH, infoldername);
	StringCchCat(szDir, MAX_PATH, TEXT("\\*"));
	// Find the first file in the directory.
	hFind = FindFirstFile(szDir, &ffd);
	if (INVALID_HANDLE_VALUE == hFind){
		//DisplayErrorBox(TEXT("FindFirstFile"));
		exit (-2);
	} 
	// List all the files in the directory with some info about them.
	do{
		if (ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY){
		}else{
			if (strstr(ffd.cFileName,filetype)!=NULL){
				filesfound++;
			}
		}
	}while (FindNextFile(hFind, &ffd) != 0);
	printf("%d *%s files found.\n",filesfound,filetype);
	return filesfound;
}

void getfilelist(char* infoldername,char** outfilelist,char* filetype){
	WIN32_FIND_DATA ffd;
	TCHAR szDir[MAX_PATH];
	size_t length_of_arg;
	HANDLE hFind = INVALID_HANDLE_VALUE;
	DWORD dwError=0;
	int i=0;
	char * tempstring;

	// Check that the input path plus 3 is not longer than MAX_PATH.
	// Three characters are for the "\*" plus NULL appended below.
	StringCchLength(infoldername, MAX_PATH, &length_of_arg);
	if (length_of_arg > (MAX_PATH - 3)){
		_tprintf(TEXT("\nDirectory path is too long.\n"));
		exit(-1);
	}

	// Prepare string for use with FindFile functions.  First, copy the
	// string to a buffer, then append '\*' to the directory name.
	StringCchCopy(szDir, MAX_PATH, infoldername);
	StringCchCat(szDir, MAX_PATH, TEXT("\\*"));

	// Find the first file in the directory.
	hFind = FindFirstFile(szDir, &ffd);
	if (INVALID_HANDLE_VALUE == hFind){
		//DisplayErrorBox(TEXT("FindFirstFile"));
		exit (-2);
	} 
   
	// List all the files in the directory with some info about them.
	do{
		if (ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY){
		//_tprintf(TEXT("  %s   <DIR>\n"), ffd.cFileName);
		}else{
			if (strstr(ffd.cFileName,filetype)!=NULL){
				tempstring=(char*)malloc(MAXCHARNUMBER*sizeof(char));
				sprintf(tempstring,ffd.cFileName);
				outfilelist[i]=tempstring;
				i++;
			}
		}
	}while (FindNextFile(hFind, &ffd) != 0);
	dwError = GetLastError();
	if (dwError != ERROR_NO_MORE_FILES){
		//DisplayErrorBox(TEXT("FindFirstFile"));
	}
	FindClose(hFind);
}
