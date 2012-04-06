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
#include "Lattice2D.h"
#include "cuda.h"
#include "Totalistic.h"
#include <map>
#include <vector>

using namespace std;

class Generations : public Totalistic{

public :
	DLLExport __device__ __host__ Generations();
	DLLExport __device__ __host__ ~Generations();


	//This is only public for access by Kernel...
	Lattice2D *lattice;

public:
	__host__ __device__ struct Cell {
	  unsigned int state;
	};

	__host__ __device__ Lattice2D* getLattice() {
		return lattice;
	}

	
	//These return a list of dynamic pointers to be put onto the GPU.
	__host__ map<void**, size_t>* getDynamicArrays();

	__host__ virtual void setLattice(AbstractLattice* newLattice);

	__host__ virtual size_t getCellSize();

	__device__  int applyFunction(void* g_data, int x, int y, int xDIM, int yDIM) { 
		
		int xAltered = x * yDIM;
		int gridLoc = x * yDIM + y;

		unsigned int* cellData = (unsigned int*)g_data;

		int state = cellData[gridLoc];

		int newState = state;

		int temp = 0;
		//generations specialism
		if (state > 1) {
			if(state < noStates - 1) {
				temp = state + 1;
				newState = setNewState(lattice,temp,state);			
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
			lattice->getNeighbourhood(neighbourhoodStates,xAltered,y,xDIM,yDIM);




			//Should probably move this code to TOTALISTIC CLASS, but it might slow it down..
			int liveCells = 0;//getLiveCellCount(neighbourhoodStates,lattice->maxBits,lattice->neighbourhoodType);

			for(int i = 0; i < lattice->getNeighbourhoodType(); ++i) {
				if(cellData[neighbourhoodStates[i]] != -1) {

			/*		int neighX = neighbourhoodStates[i] / xDIM;
					int neighY = neighbourhoodStates[i] % xDIM;
					
					neighY = neighY - (blockIdx.y * blockDim.y);
					neighX = neighX - (blockIdx.x * blockDim.x);*/

					//And back to our flat shared array..
					if((cellData[neighbourhoodStates[i]] & lattice->getMaxBits()) == 1) //This cell's state is alive.
						++liveCells;
				}
			}


			//we only care about neighbours when we know we're in a ready state
			//int liveCells = Totalistic::getLiveCellCount(neighbourhoodStates,lattice->maxBits,lattice->neighbourhoodType);
	
			if(state == 1) {
				
				for (int i = 0; i < surviveSize; i++) {
					if (liveCells == surviveNo[i]) newState = setNewState(lattice,1,state);
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
		}

		//Potential bug here, could writing corrupt our data ??
		cellData[gridLoc] = newState;

	}
};


#endif
