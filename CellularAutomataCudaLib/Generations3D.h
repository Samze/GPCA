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
#include "cuda_runtime.h"
#include <map>
#include <vector>

using namespace std;

class Generations3D : public AbstractCellularAutomata {

public :
	DLLExport __device__ __host__ Generations3D() {}
	DLLExport __device__ __host__ ~Generations3D() {}
	
    int* surviveNo;
    int  surviveSize;

	int* bornNo;
    int bornSize;
	
	Abstract3DCA *lattice;
	
	__device__ __host__ void setSurviveNo(int* list, int size) {
		surviveNo = list;
		surviveSize = size;
	}

	__device__ __host__ void setBornNo(int* list, int size) {
		bornNo = list;
		bornSize = size;
	}

	__host__ virtual size_t getCellSize() {
		return sizeof(unsigned int);
	}

	//These return a list of dynamic pointers to be put onto the GPU.
	__host__ map<void**, size_t>* getDynamicArrays() {
		
		map<void**, size_t>* newMap = new map<void**, size_t>();

		size_t gridMemSize = lattice->DIM * lattice->DIM * lattice->DIM * sizeof(unsigned int);

		newMap->insert(make_pair((void**)&lattice->pFlatGrid, gridMemSize));
		newMap->insert(make_pair((void**)&bornNo,sizeof(int) * bornSize));
		newMap->insert(make_pair((void**)&surviveNo,sizeof(int) * surviveSize));


		return newMap;
	}


	__host__ __device__ virtual AbstractLattice* getLattice() { return lattice;}

	__device__  int applyFunction(void* g_data, int x, int y, int z, int xDIM) { 

		int xAltered = x * xDIM;
		int zAltered = z * xDIM * xDIM;

		unsigned int* cellData = (unsigned int*)g_data;
		
		int gridLoc = xAltered + zAltered + y;

		int state = cellData[gridLoc];
	//	//We want know about neighbours even if we're not using them to set the next state, this is 
	//	//so they can not be rendered by the viewer. To speed up the processing, move this to the else

		int neighbourhoodStates[26];//= {-1};
	//
	//	//set as -1 by default.
		//TODO Add unrolling here!
		for(int i = 0; i < 26; i++) {
			neighbourhoodStates[i] = -1; 
		}


	//	//Populate neighbours states.
		lattice->getNeighbourhood(neighbourhoodStates,xAltered,y,zAltered,xDIM);


		//populate our neighbours with their cell state values 
		//TODO Add unrolling here!
		for(int i= 0; i <26; i++) {
			if(neighbourhoodStates[i] != -1) {
				neighbourhoodStates[i] = cellData[neighbourhoodStates[i]];
			}
		}

		unsigned int liveCells = Totalistic::getLiveCellCount(neighbourhoodStates,lattice->maxBits,lattice->neighbourhoodType);

	//	//int liveCells = getNeighbourhood(g_data, xAltered, y, zAltered, xDIM);

	//	//This is for culling of cubes surrounded on all sides.
		
		//lattice->neighbourCount[xAltered + y + zAltered] = liveCells;
		

		unsigned int newState = state;

		if (state > 1) {
			if(state < noStates - 1) {
				int temp = state + 1;
				newState = setNewState(lattice,temp,state);
			}
		}
		else if(state == 1){
			for (int i = 0; i < surviveSize; i++) {
				if (liveCells == surviveNo[i]) {
					newState = setNewState(lattice,1,state);
					//Should break early here...
				}
			}
		}
		else if(state == 0) {
			for (int i = 0; i < bornSize; ++i) {		
				if (liveCells == bornNo[i]) newState = setNewState(lattice,1,state);
			}
		}

		if (state == 1 && newState == state) {
				if (state < noStates - 1) { //This guards against 2 state generations
					newState = setNewState(lattice,2,state);
				}
		}
		

		//Potential bug here, could writing corrupt our data ??
		cellData[gridLoc] = newState;
	}
};


#endif
