/***********************************************************************************************************************
This file is part of "SIFT-GPU-Thesis" project.
 * Copyright (C) 2015 {Christos Papazachariou} <{christospapazachariou@gmail.com}>
 * University of Piraeus- MSc program: Advanced Information Systems ("Embedded System Technologies" branch).
 * Partial or otherwise use of this project and/or source code is permitted only for educational and academic purposes.
 * Partial or otherwise use of this project and/or source code for commercial or militay applications is prohibited.
***********************************************************************************************************************/

#ifndef _MYTIMER_H_
#define _MYTIMER_H_

//GLOBAL MYTIMER VARS AND DEFS
#define MAX_TIMETABLE_ENTRIES 5000	//static array dimensions to avoid global malloc
#define SEPARATOR_CHARS 80

extern int timeindex;
extern int64 timerstart;

extern int displayflag;	//when 0 do not display demo/debug pics
extern int saveflag;
extern int showmemoryflag;
extern int showtimeflag;

typedef struct timestamp_t
{
	int64 ticks;
	char* timelabel;
} timestamp;

void logtime(char* currentlabel);
void showtimes();
void savetimes(char* imagename);

extern char separator[SEPARATOR_CHARS];
void setSeparator(char input);

extern timestamp timetable[MAX_TIMETABLE_ENTRIES];
				extern timestamp timetable2[MAX_TIMETABLE_ENTRIES];

		void openShow3(unsigned char* input, int inH, int inW,char* debuglabel);
			void logtime2(char* currentlabel);
			void savetimes2(char* imagename);
			void showtimes2();

#endif
