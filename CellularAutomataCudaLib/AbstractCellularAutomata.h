/*	GPCA - A Cellular Automata library powered by CUDA. 
    Copyright (C) 2011  Sam Gunaratne University of Plymouth

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef ABSTRACT_CELLULAR_AUTOMATA_H
#define ABSTRACT_CELLULAR_AUTOMATA_H

#include "device_launch_parameters.h"

#define DLLExport __declspec(dllexport)

class AbstractCellularAutomata
{

public:
	DLLExport AbstractCellularAutomata(void) {}
	DLLExport virtual ~AbstractCellularAutomata(void) {}
	
	DLLExport __host__ void setStates(unsigned int states);
	DLLExport __host__ int getNoStates() { return noStates;}
	__host__ int getNoBits() { return noBits;}
	
	unsigned int *pFlatGrid;
	unsigned int DIM;


protected:
	int noStates;
	int noBits;
	int maxBits;


	//This next line should be here to provide 'proper' virtual inheritence, sadly it is only supported on CUDA sm_2x architecture.
	//__device__ __host__ int applyFunction(int*,int,int,int) {
	//	return 3;
	//}

};


#endif