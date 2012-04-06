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

#include "device_launch_parameters.h"
#include "AbstractCellularAutomata.h"
#include "cuda.h"

class Totalistic : public AbstractCellularAutomata{

public:
	DLLExport __device__  __host__ Totalistic();
	DLLExport __device__  __host__ virtual ~Totalistic();
	 
	DLLExport __host__ virtual void setStates(unsigned int states);

	DLLExport __host__ int getNoStates() { return noStates;}

	__host__ __device__ int setNewState(AbstractLattice* lattice, int newState, int oldState) {
		return oldState | (newState << lattice->getNoBits());
	}


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


	__device__ __host__  static unsigned int getLiveCellCount(int* neighbourhoodStates, int maxBits, int neighbourType) {

		unsigned int numLiveCells =0;

		for(int i = 0; i < neighbourType; ++i) {
			if(neighbourhoodStates[i] != -1) 
				if((neighbourhoodStates[i] & maxBits) == 1) //This cell's state is alive.
					++numLiveCells;
		}

		return numLiveCells;
	}

	
protected:
	int noStates;
};

