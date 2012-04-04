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
#include "Lattice2D.h"
#include "Totalistic.h"
#include "cuda.h"
#include <map>
#include <vector>

using namespace std;

class OuterTotalistic : public Totalistic {

public :
	DLLExport __device__ __host__ OuterTotalistic() {}
	DLLExport __device__ __host__ ~OuterTotalistic() {delete lattice;}

	__host__ __device__ struct Cell {
	  unsigned int state;
	};

	Lattice2D *lattice;
	
	__host__ __device__ virtual AbstractLattice* getLattice() { return lattice;}
	
	//These return a list of dynamic pointers to be put onto the GPU.
	__host__ map<void**, size_t>* getDynamicArrays() {
		
		map<void**, size_t>* newMap = new map<void**, size_t>();

		size_t gridMemSize = lattice->xDIM * lattice->yDIM * sizeof(unsigned int);

		newMap->insert(make_pair((void**)&lattice->pFlatGrid, gridMemSize));
		newMap->insert(make_pair((void**)&bornNo,sizeof(int) * bornSize));
		newMap->insert(make_pair((void**)&surviveNo,sizeof(int) * surviveSize));

		return newMap;
	}

	__host__ virtual size_t getCellSize() {
		return sizeof(unsigned int);
	}

	//TODO move this to .cpp
	__host__ virtual void setLattice(AbstractLattice* newLattice) {

		if(newLattice == lattice)
			return;

		Lattice2D* new2DLattice = dynamic_cast<Lattice2D*>(newLattice);

		lattice = new2DLattice;
		
	} 

	__device__ __host__ int applyFunction(void* g_data, int x, int y, int xDIM,int yDIM) { 
		
		int xAltered = x * yDIM;
		int gridLoc = x * yDIM + y;

		unsigned int* cellData = (unsigned int*)g_data;

		int state = cellData[gridLoc];

		int newState = state;
		
		int neighbourhoodStates[8];
	
		//set as -1 by default.
		for(int i = 0; i < 8; i++) {
			neighbourhoodStates[i] = -1; 
		}

		lattice->getNeighbourhood(neighbourhoodStates,xAltered,y,xDIM,yDIM);

		//Should probably move this code to TOTALISTIC CLASS, but it might slow it down..
		int liveCells = 0;//getLiveCellCount(neighbourhoodStates,lattice->maxBits,lattice->neighbourhoodType);

		for(int i = 0; i < lattice->neighbourhoodType; ++i) {
			if(cellData[neighbourhoodStates[i]] != -1) {
				if((cellData[neighbourhoodStates[i]] & lattice->maxBits) == 1) //This cell's state is alive.
					++liveCells;
			}
		}

		for (int i = 0; i < surviveSize; i++) {
			if (state == 1 && liveCells == surviveNo[i]) newState = setNewState(lattice,1,state);
		}

		for (int i = 0; i < bornSize; ++i) {		
			if (state == 0 && liveCells == bornNo[i]) newState = setNewState(lattice,1,state);
		}

		cellData[gridLoc] = newState;
	}
};


#endif
