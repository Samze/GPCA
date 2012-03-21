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
#pragma once

#include <device_launch_parameters.h>
#include <device_functions.h>
#include "abstractcellularautomata.h"
#include "Abstract2DCA.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include <map>
#include <vector>

using namespace std;

class SCIARA2 :
	public AbstractCellularAutomata
{
public:
	DLLExport __device__ __host__ SCIARA2(void);
	DLLExport __device__ __host__  ~SCIARA2(void);

	Abstract2DCA *lattice;

	//For now this must be signed to cope with -1 (no neighbour values)
	__host__ __device__ struct Cell {
	  float altitude;
	  float thickness;
	  float outflow[4];
	};
	
	__host__ virtual size_t getCellSize() {
		return sizeof(Cell);
	}

	//TODO move this to .cpp
	__host__ virtual void setLattice(AbstractLattice* newLattice) {

		if(newLattice == lattice)
			return;

		Abstract2DCA* new2DLattice = dynamic_cast<Abstract2DCA*>(newLattice);

		lattice = new2DLattice;
		
	} 

	__host__ __device__ virtual AbstractLattice* getLattice() { return lattice;}


	__host__ map<void**, size_t>* getDynamicArrays() {
		
		map<void**, size_t>* newMap = new map<void**, size_t>();

		size_t gridMemSize = lattice->xDIM * lattice->yDIM * sizeof(unsigned int);

		newMap->insert(make_pair((void**)&lattice->pFlatGrid, gridMemSize));

		return newMap;
	}

	/* State information.

	z = 0-1000;
	h = 0-100;
	f[4] = 0-100 (400)

	max = 1000 * 100 * 800;
	*/
	__device__  int applyFunction(void* g_data, int x, int y, int xDIM, int yDIM) { 

		int xAltered = x * yDIM;
		int gridLoc = x * yDIM + y;

		Cell* cellGrid = (Cell*)g_data;

		Cell centerCell = cellGrid[gridLoc]; 

		Cell neighs[4];

		int neighbourhoodStates[4];

	//	//set as -1 by default.
		for(int i = 0; i < 4; i++) {
			neighbourhoodStates[i] = -1; 
		}

		lattice->getNeighbourhood(neighbourhoodStates,xAltered,y,xDIM,yDIM);

		//Populate neighbours
		for(int i = 0; i < 4; i++) {
			int address = neighbourhoodStates[i];
			
			if (address != -1) {
				neighs[i] =  cellGrid[address];
			}
			else {
				neighs[i].altitude = 10000; //set to max height
				neighs[i].thickness = 0;
				neighs[i].outflow[0] = 0;
				neighs[i].outflow[1] = 0;
				neighs[i].outflow[2] = 0;
				neighs[i].outflow[3] = 0;
			}
		}


		//Update outflows

		bool elim[5] = {false,false,false,false,false};
		//Mobile part of center
		float m;

		float Z[5];

		float average;

		int k,i;
		bool again;
	
		m = centerCell.thickness;

		Z[0] = centerCell.altitude;

		for(int i = 1; i < 5; i++) {
			Z[i] = neighs[i-1].altitude + neighs[i-1].thickness;
		}


		do {
			again = false;
			k = 0;
			average = m;

			for(i = 0;  i<5 ; i++) {
				if ( elim[i] == false) {
					average += Z[i];
					k++;
				}
			}
			average = average/k;

			for(i=0;i<5;i++) {
				if ((average <= Z[i]) && elim[i] == false) {
					elim[i] = true;
					again = true;
				}
			}
		}
		while(again);

		for (i = 1; i < 5; i++) {
			if (elim[i] == false) {
				centerCell.outflow[i-1] = (average - Z[i]) * 0.7;
			}
			else {
				centerCell.outflow[i-1] = 0;
			}
		}

		//updateCell
		cellGrid[gridLoc] = centerCell;

	}

	__device__  int computethickness(void* g_data, int x, int y, int xDIM, int yDIM) { 
		
		int xAltered = x * yDIM;
		int gridLoc = x * yDIM + y;

		Cell* cellGrid = (Cell*)g_data;


		//cuda sm1.1 does not support recursion, shame.
		Cell centerCell = cellGrid[gridLoc]; 

		Cell neighs[4];

		int neighbourhoodStates[4];
	//
	//	//set as -1 by default.
		for(int i = 0; i < 4; i++) {
			neighbourhoodStates[i] = -1; 
		}

		lattice->getNeighbourhood(neighbourhoodStates,xAltered,y,xDIM,yDIM);

		//Populate neighbours
		for(int i = 0; i < 4; i++) {
			int address = neighbourhoodStates[i];
			
			if (address != -1) {
				neighs[i] =  cellGrid[address];
			}
			else {
				neighs[i].altitude = 10000; //set to max height
				neighs[i].thickness = 0;
				neighs[i].outflow[0] = 0;
				neighs[i].outflow[1] = 0;
				neighs[i].outflow[2] = 0;
				neighs[i].outflow[3] = 0;
			}
		}

		
		int i;

		//Used for debugging
		float outflows = 0;

		float new_thickness = centerCell.thickness;

		for(i=0; i < 4; i++) {
			new_thickness = new_thickness - centerCell.outflow[i] + neighs[i].outflow[3-i];
			outflows += centerCell.outflow[i];
		}

		centerCell.thickness = new_thickness;
		//centerCell.thickness = 100;


		//updateCell
		cellGrid[gridLoc] = centerCell;
	}

};

