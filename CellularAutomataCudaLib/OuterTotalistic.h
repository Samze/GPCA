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

#ifndef OUTER_TOTALISTIC_H_
#define OUTER_TOTALISTIC_H_

#include "device_launch_parameters.h"
#include "AbstractCellularAutomata.h"
#include "Abstract2DCA.h"
#include "Totalistic.h"

class OuterTotalistic : public AbstractCellularAutomata {

public :
	DLLExport __device__ __host__ OuterTotalistic() {}
	DLLExport __device__ __host__ ~OuterTotalistic() {}

	__host__ __device__ struct Cell {
	  unsigned int state;
	};
	
	int* surviveNo;
    int  surviveSize;

	int* bornNo;
    int bornSize;
	
	Abstract2DCA *lattice;

	__host__ virtual size_t getCellSize() {
		return sizeof(unsigned int);
	}

	__device__ __host__ int applyFunction(void* g_data, int x, int y, int xDIM) { 
		
	//	int gridLoc = x * xDIM + y;
	//	int state = g_data[gridLoc];
	//	
	//	int neighbourhoodStates[8];
	//
	//	//set as -1 by default.
	//	for(int i = 0; i < 8; i++) {
	//		neighbourhoodStates[i] = -1; 
	//	}

	//	//lattice->getNeighbourhood(neighbourhoodStates,g_data,gridLoc);

	//	int liveCells = Totalistic::getLiveCellCount(neighbourhoodStates,lattice->maxBits,lattice->neighbourhoodType);

	//		for (int i = 0; i < surviveSize; i++) {
	//			if (state == 1 && liveCells == surviveNo[i]) return setNewState(lattice,1,state);
	//		}
	//		
	//		for (int i = 0; i < bornSize; ++i) {		
	//			if (state == 0 && liveCells == bornNo[i]) return setNewState(lattice,1,state);
	//		}

	//	return state; 
	}
};


#endif
