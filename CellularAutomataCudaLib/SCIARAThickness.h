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
#include "Lattice2D.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include <map>
#include <vector>

using namespace std;

class SCIARAThickness:
	public AbstractCellularAutomata
{
public:
	DLLExport __device__ __host__ SCIARAThickness(void);
	DLLExport __device__ __host__  ~SCIARAThickness(void);

	//This is only public for access by Kernel...
	Lattice2D *lattice;

public:

	//For now this must be signed to cope with -1 (no neighbour values)
	__host__ __device__ struct Cell {
	  float altitude;
	  float thickness;
	  float outflow[4];
	};
	__host__ virtual void setLattice(AbstractLattice* newLattice);
	__host__ map<void**, size_t>* getDynamicArrays();

	__host__ __device__ Lattice2D* getLattice() { 
		return lattice;
	}

	__host__ __device__ int applyFunction(void* g_data, int x, int y, int xDIM, int yDIM) { 
		
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

		lattice->getNeighbourhood(neighbourhoodStates,x,y,xDIM,yDIM);

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

		//centerCell.thickness = new_thickness;

		//updateCell
		cellGrid[gridLoc].thickness = new_thickness;

		return 0;
	}
};

