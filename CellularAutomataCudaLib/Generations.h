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

#ifndef GENERATIONS_H
#define GENERATIONS_H

#include "device_launch_parameters.h"
#include "AbstractCellularAutomata.h"
#include "Abstract2DCA.h"
#include "cuda.h"
#include "Totalistic.h"

class Generations : public AbstractCellularAutomata{

public :
	DLLExport __device__ __host__ Generations() {}
	DLLExport __device__ __host__ ~Generations() {}

    int* surviveNo;
    int  surviveSize;
	
	__device__ __host__ void setSurviveNo(int* list, int size) {
		surviveNo = list;
		surviveSize = size;
	}

	__device__ __host__ void setBornNo(int* list, int size) {
		bornNo = list;
		bornSize = size;
	}
	
	int* bornNo;
    int bornSize;

	Abstract2DCA *lattice;
	
	__host__ __device__ virtual AbstractLattice* getLattice() { return lattice;}

	__device__  int applyFunction(unsigned int* g_data, int x, int y, int xDIM) { 
		
		int gridLoc = x * xDIM + y;
		int state = g_data[gridLoc];
		int temp = 0;

		//generations specialism
		if (state > 1) {
			if(state >= noStates - 1) {
				//reset this state next go
				return state;
			}
			else {
				temp = state + 1;
				return setNewState(lattice,temp,state);			
			}
		}
		else {
			//This is signed so we can default to -1, this shows NO neighbour, a result of 0 means a neighbour who's state is zero
			//This may cause complications as states are stored in unsigned ints...
			int neighbourhoodStates[8];
	
			//set as -1 by default.
			for(int i = 0; i < 8; i++) {
				neighbourhoodStates[i] = -1; 
			}

			//Populate neighbours states.
			lattice->getNeighbourhood(neighbourhoodStates,g_data,gridLoc);

			//we only care about neighbours when we know we're in a ready state
			int liveCells = Totalistic::getLiveCellCount(neighbourhoodStates,lattice->maxBits,lattice->neighbourhoodType);
	
			for (int i = 0; i < surviveSize; i++) {
				if (state == 1 && liveCells == surviveNo[i]) return setNewState(lattice,1,state);
			}
			
			for (int i = 0; i < bornSize; ++i) {		
				if (state == 0 && liveCells == bornNo[i]) return setNewState(lattice,1,state);
			}
			
			if (state == 1) {
				if (state < noStates - 1) { //This guards against 2 state generations
					return setNewState(lattice,2,state);
				}
			}
		}


		return state;

	}

	//DLLExport virtual void setStates(unsigned int states) {

	//	noStates = states;

	//	//calculate how many bits are needed to hold a states
	//	//we need to minus one to properly reflect the fact that 1 bit can hold 2 states
	//	// 3 bits can hold 8 states etc.

	//	states = states - 1;

	//	lattice->noBits = 0;
	//	while (states != 0) { 
	//		states = states >> 1; 
	//		++lattice->noBits;
	//	}

	//	lattice->maxBits = 1;

	//	for (int i = 1; i < lattice->noBits; i++) {
	//		lattice->maxBits = (lattice->maxBits << 1) + 1;
	//	}
	//}

};


#endif
