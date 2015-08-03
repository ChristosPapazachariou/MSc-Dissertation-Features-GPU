/***********************************************************************************************************************
This file is part of "SIFT-GPU-Thesis" project.
 * Copyright (C) 2015 {Christos Papazachariou} <{christospapazachariou@gmail.com}>
 * University of Piraeus- MSc program: Advanced Information Systems ("Embedded System Technologies" branch).
 * Partial or otherwise use of this project and/or source code is permitted only for educational and academic purposes.
 * Partial or otherwise use of this project and/or source code for commercial or militay applications is prohibited.
***********************************************************************************************************************/

#include "sift.h"

#include "ioocv.h"
#include "MYtimer.h"

extern int timeindex=0;
extern int64 timerstart=0;
extern int displayflag=0;	//when 0 do not display demo/debug pics
extern int saveflag=0;
extern int showmemoryflag=0;
extern int showtimeflag=0;

timestamp timetable[MAX_TIMETABLE_ENTRIES];//={0,'n'};
		timestamp timetable2[MAX_TIMETABLE_ENTRIES];//={0,'n'};
extern char separator[SEPARATOR_CHARS]={0};

void setSeparator(char input){
	int i=0;
	for (i=0;i<SEPARATOR_CHARS;i++){
		separator[i]=input;
	}
}

void logtime(char* currentlabel){	
	timetable[timeindex].ticks=cvGetTickCount()-timerstart;
	timetable[timeindex].timelabel=currentlabel;
	//printf("timing %03d logged: %s : %06lld ticks\n",timeindex, timetable[timeindex].timelabel, timetable[timeindex].ticks);
	timeindex++;
}



void showtimes(){
	int i=0;
	double mils=0;
	double freq=cvGetTickFrequency()*1000;
	if (showtimeflag!=0){
		printf("%s\nDisplaying timelog...\n",separator);
		for (i=0;i<timeindex;i++){
			//mils=timetable[i].ticks/freq;
			mils=(double)(timetable[i].ticks/freq);
			printf("entry %03d:\t%08.3lf ms\t%06lld ticks\t %s\n",i,mils,timetable[i].ticks,timetable[i].timelabel);
		}
		printf("Timelog end.\n%s\n",separator);
	}
}

void savetimes(char* imagename){
	FILE * logfile;
	FILE * logarray;
	int i;
	double mils=0;
	char  logfilename[200];
	char  logarrayname[200];
	double freq=cvGetTickFrequency()*1000;
	sprintf(logfilename,"producedTimelogs/timelog_%s.txt",imagename);
	sprintf(logarrayname,"producedTimelogs/timearray_%s.txt",imagename);
	//logfilename="producedTimelogs/timelog.txt";
	printf("Saving timelog to file: %s\n",logfilename);
	printf("Saving timelog array to file: %s\n",logarrayname);
	logfile=fopen(logfilename, "w+");
	logarray=fopen(logarrayname, "w+");
	for (i=0;i<timeindex;i++){
		//mils=timetable[i].ticks-timerstart/freq;
		mils=(double)(timetable[i].ticks/freq);
		fprintf(logfile,"entry %03d:\t%08.3lf ms\t%06lld ticks\t %s\n",i,mils,timetable[i].ticks,timetable[i].timelabel);
		fprintf(logarray,"%s\t%08.3lf\n",timetable[i].timelabel,mils);
	}
	printf("Timelog saved.\n");
	fclose (logfile);
	fclose (logarray);
}

void savetimes2(char* imagename){
	FILE * logfile;
	FILE * logarray;
	int i;
	double mils=0;
	char  logfilename[200];
	char  logarrayname[200];
	double freq=cvGetTickFrequency()*1000;
	sprintf(logfilename,"producedTimelogs/2timelog_%s.txt",imagename);
	sprintf(logarrayname,"producedTimelogs/2timearray_%s.txt",imagename);
	//logfilename="producedTimelogs/timelog.txt";
	printf("Saving timelog to file: %s\n",logfilename);
	printf("Saving timelog array to file: %s\n",logarrayname);
	logfile=fopen(logfilename, "w+");
	logarray=fopen(logarrayname, "w+");
	for (i=0;i<timeindex;i++){
		//mils=timetable[i].ticks-timerstart/freq;
		mils=(double)(timetable2[i].ticks/freq);
		fprintf(logfile,"entry %03d:\t%08.3lf ms\t%06lld ticks\t %s\n",i,mils,timetable2[i].ticks,timetable2[i].timelabel);
		fprintf(logarray,"%s\t%08.3lf\n",timetable2[i].timelabel,mils);
	}
	printf("Timelog saved.\n");
	fclose (logfile);
	fclose (logarray);
}

void logtime2(char* currentlabel){	
	timetable2[timeindex].ticks=cvGetTickCount()-timerstart;
	timetable2[timeindex].timelabel=currentlabel;
	timeindex++;
}

void showtimes2(){
	int i=0;
	double mils=0;
	double freq=cvGetTickFrequency()*1000;
	if (showtimeflag!=0){	
		printf("%s\nDisplaying timelog...\n",separator);
		for (i=0;i<timeindex;i++){
			mils=(double)(timetable2[i].ticks/freq);
			printf("entry %03d:\t%08.3lf ms\t%06lld ticks\t %s\n",i,mils,timetable2[i].ticks,timetable2[i].timelabel);
		}
		printf("Timelog end.\n%s\n",separator);
	}
}

//this function is in this file because this file and its header can be safely referenced from cuda functions. ioocv.h and ioocv.c are not extern "C" safe as it is
		void openShow3(unsigned char* input, int inH, int inW, char* debuglabel){
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

