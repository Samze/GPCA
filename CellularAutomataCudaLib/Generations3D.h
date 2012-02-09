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

#ifndef GENERATIONS3D_H
#define GENERATIONS3D_H

#include "device_launch_parameters.h"
#include "AbstractCellularAutomata.h"
#include "Abstract3DCA.h"
#include "cuda.h"

class Generations3D : public AbstractCellularAutomata {

public :
	DLLExport __device__ __host__ Generations3D() {}
	DLLExport __device__ __host__ ~Generations3D() {}
	
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

	Abstract3DCA *lattice;
	
	__host__ __device__ virtual AbstractLattice* getLattice() { return lattice;}

	__device__  int applyFunction(unsigned int* g_data, int x, int y, int z, int xDIM) { 
		
		int xAltered = x * xDIM;
		int zAltered = z * xDIM * xDIM;

		int gridLoc = xAltered + zAltered + y;

		int state = g_data[gridLoc];
		int temp = 0;

		//We want know about neighbours even if we're not using them to set the next state, this is 
		//so they can not be rendered by the viewer. To speed up the processing, move this to the else

		int neighbourhoodStates[26];
	
		//set as -1 by default.
		for(int i = 0; i < 26; i++) {
			neighbourhoodStates[i] = -1; 
		}

		//Populate neighbours states.
		//lattice->getNeighbourhood(neighbourhoodStates,xAltered,y,zAltered,g_data,gridLoc);
		lattice->getNeighbourhood(neighbourhoodStates,g_data,gridLoc);

		//we only care about neighbours when we know we're in a ready state
		//int liveCells =  getNeighbourhood(g_data, x * xDIM, y, xDIM, neighbourhoodType);
		int liveCells = Totalistic::getLiveCellCount(neighbourhoodStates,lattice->maxBits,lattice->neighbourhoodType);

		//int liveCells = getNeighbourhood(g_data, xAltered, y, zAltered, xDIM);

		//This is for culling of cubes surrounded on all sides.
		lattice->neighbourCount[xAltered + y + zAltered] = liveCells;

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
};


#endif
