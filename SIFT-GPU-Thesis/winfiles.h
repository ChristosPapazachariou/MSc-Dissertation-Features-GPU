/***********************************************************************************************************************
This file is part of "SIFT-GPU-Thesis" project.
 * Copyright (C) 2015 {Christos Papazachariou} <{christospapazachariou@gmail.com}>
 * University of Piraeus- MSc program: Advanced Information Systems ("Embedded System Technologies" branch).
 * Partial or otherwise use of this project and/or source code is permitted only for educational and academic purposes.
 * Partial or otherwise use of this project and/or source code for commercial or militay applications is prohibited.
***********************************************************************************************************************/

#ifndef WINFILES_H
#define WINFILES_H

#define MAXCHARNUMBER 100

int countfiles(char* infoldername, char* filetype);
void getfilelist(char* infoldername, char** outfilelist, char*filetype);

#endif