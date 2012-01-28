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


class Abstract2DCA : public AbstractCellularAutomata
{

public:
	DLLExport Abstract2DCA(void);
	DLLExport virtual ~Abstract2DCA(void); //This was virtual...????

	//Always better to use constants than defines (effective C++)
	//Using 8 and 4 as the number of neighbours..
	static const int MOORE = 8;
	static const int VON_NEUMANN = 4;
	
	int neighbourhoodType;

    int* surviveNo;
    int  surviveSize;

	int* bornNo;
    int bornSize;


	__device__ __host__ void setSurviveNo(int* list, int size) {
		surviveNo = list;
		surviveSize = size;
	}

	__device__ __host__ void setBornNo(int* list, int size) {
		bornNo = list;
		bornSize = size;
	}
	
	__device__ __host__ void getNeighbourhood(int* neighbourStates, unsigned int* g_data, int x, int y, int xDIM) {

		switch(neighbourhoodType) {
		case MOORE:
			getMooresNeighbourhood(neighbourStates,g_data,x,y,xDIM);
			break;
		case VON_NEUMANN:
			getVonNeumannNeighbourhood(neighbourStates,g_data,x,y,xDIM);
			break;
		default:
			break;
		}
	}


	//probably a much better way to figure out the moores neighbourhood, populates a max of 8 neighbours
	__device__ __host__ void getMooresNeighbourhood(int* neighbours, unsigned int* g_data, int x, int y, int xDIM) {

		// [-1,-1]
		if (x != 0 && y != 0)
			neighbours[0] = g_data[x - xDIM + y - 1];

		// [0,-1]
		if ( y != 0)
			neighbours[1] = g_data[x + y - 1];

		// [1,-1]
		if (x != xDIM - 1 && y != 0 )
			neighbours[2] = g_data[x + xDIM + y - 1];

		// [-1,0]
		if (x != 0)
			neighbours[3] = g_data[x - xDIM + y];

		// [1,0]
		if (x != xDIM - 1)
			neighbours[4] =  g_data[x + xDIM + y];

		// [-1,1]
		if (x != 0 && y != xDIM -1 )
			neighbours[5] = g_data[x - xDIM + y + 1];

		// [0,1]
		if (y != xDIM -1 )
			neighbours[6] = g_data[x + y + 1];

		// [1,1]
		if (x != xDIM -1 && y != xDIM - 1 )
			neighbours[7] = g_data[x + xDIM + y + 1];

	}
	//probably a much better way to figure out the moores neighbourhood, populates a max of 8 neighbours
	__device__ __host__ void getVonNeumannNeighbourhood(int* neighbours, unsigned int* g_data, int x, int y, int xDIM) {

		// [0,-1]
		if ( y != 0)
			neighbours[0] = g_data[x + y - 1];

		// [-1,0]
		if (x != 0)
			neighbours[1] = g_data[x - xDIM + y];

		// [1,0]
		if (x != xDIM - 1)
			neighbours[2] =  g_data[x + xDIM + y];

		// [0,1]
		if (y != xDIM -1 )
			neighbours[3] = g_data[x + y + 1];
	}
};

