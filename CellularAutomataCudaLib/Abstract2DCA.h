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
#include "AbstractLattice.h"


class Abstract2DCA : public AbstractLattice
{

public:
	DLLExport Abstract2DCA(void);
	DLLExport Abstract2DCA(int,int); //random data
	DLLExport Abstract2DCA(void*, int);

	DLLExport virtual ~Abstract2DCA(void); //This was virtual, but this class isn't abstract..?
	
	//Currently not used..
	__host__ virtual size_t size() const { return sizeof(this); }

	//Always better to use constants than defines (effective C++)
	//Using 8 and 4 as the number of neighbours..
	static const int MOORE = 8;
	static const int VON_NEUMANN = 4;

	__device__ __host__ void getNeighbourhood(int* neighbourStates, int x, int y, int DIM) {

		switch(neighbourhoodType) {
		case MOORE:
			getMooresNeighbourhood(neighbourStates,x,y,DIM);
			break;
		case VON_NEUMANN:
			getVonNeumannNeighbourhood(neighbourStates,x,y,DIM);
			break;
		default:
			break;
		}
	}

private:

	//probably a much better way to figure out the moores neighbourhood, populates a max of 8 neighbours
	__device__ __host__ void getMooresNeighbourhood(int* neighbours,int x, int y, int DIM) {
		

		bool xBounds = (x / DIM) < DIM -1 ;

		// [-1,-1]
		if (x != 0 && y != 0)
			neighbours[0] = x - DIM + y - 1;

		// [0,-1]
		if ( y != 0)
			neighbours[1] = x + y - 1;

		// [1,-1]
		if (xBounds && y != 0 )
			neighbours[2] = x + DIM + y - 1;

		// [-1,0]
		if (x != 0)
			neighbours[3] = x - DIM + y;

		// [1,0]
		if (xBounds)
			neighbours[4] = x + DIM + y;

		// [-1,1]
		if (x != 0 && y != DIM - 1)
			neighbours[5] = x - DIM + y + 1;

		// [0,1]
		if (y != DIM -1 )
			neighbours[6] = x + y + 1;

		// [1,1]
		if (xBounds && y != DIM - 1 )
			neighbours[7] = x + DIM + y + 1;

	}

	//Populates a max of 4 neighbours
	__device__ __host__ void getVonNeumannNeighbourhood(int* neighbours, int x, int y, int DIM) {
		
		bool xBounds = (x / DIM) < DIM -1 ;

		// [0,-1]
		if ( y != 0){
			neighbours[0] = x + y - 1;
		}

		// [-1,0]
		if (x != 0){
			neighbours[1] = x - DIM + y;
		}

		// [1,0]
		if (xBounds){
			neighbours[2] = x + DIM + y;
		}

		// [0,1]
		if (y != DIM - 1) {
			neighbours[3] = x + y + 1;
		}
	}
};


