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
#include "Totalistic.h"
#include "Lattice3D.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include <map>
#include <vector>

using namespace std;

class Generations3D : public Totalistic {

public :
	DLLExport __device__ __host__ Generations3D();
	DLLExport __device__ __host__ ~Generations3D();


	//This is only public for access by Kernel...
	Lattice3D *lattice;

public:
	__host__ virtual size_t getCellSize();
	__host__ map<void**, size_t>* Generations3D::getDynamicArrays();
	__host__ virtual void setLattice(AbstractLattice* newLattice);
	
	__host__ __device__ Lattice3D* getLattice() { return lattice;}

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

		unsigned int liveCells = Totalistic::getLiveCellCount(neighbourhoodStates,lattice->getMaxBits(),lattice->getNeighbourhoodType());

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
