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
	__device__ struct NotACell {
	  int foo;
	  int thickness;
	  int outflow[4];
	};

public:
	DLLExport Abstract2DCA(void);
	DLLExport Abstract2DCA(int,int); //random data
	DLLExport Abstract2DCA(unsigned int*, int);

	DLLExport virtual ~Abstract2DCA(void); //This was virtual...????
	
	__host__ virtual size_t size() const { return sizeof(this); }

	//Always better to use constants than defines (effective C++)
	//Using 8 and 4 as the number of neighbours..
	static const int MOORE = 8;
	static const int VON_NEUMANN = 4;

	NotACell cell;
	

	__device__ __host__ void getNeighbourhood(int* neighbourStates, unsigned int* g_data, int gLocation){

		int x = gLocation / DIM;
		int y = gLocation % DIM;
		
		//this looks odd after the previous lines, however when diving we lose the remainder. so xAltered != gLocation
		int xAltered = x * DIM;

		switch(neighbourhoodType) {
		case MOORE:
			getMooresNeighbourhood(neighbourStates,g_data,xAltered,y,DIM);
			break;
		case VON_NEUMANN:
			getVonNeumannNeighbourhood(neighbourStates,g_data,xAltered,y,DIM);
			break;
		default:
			break;
		}
	}

	__device__ __host__ void getNeighbourhood(int* neighbourStates, unsigned int* g_data, int x, int y) {

		switch(neighbourhoodType) {
		case MOORE:
			getMooresNeighbourhood(neighbourStates,g_data,x,y,DIM);
			break;
		case VON_NEUMANN:
			getVonNeumannNeighbourhood(neighbourStates,g_data,x,y,DIM);
			break;
		default:
			break;
		}
	}

	 __device__ void* test() {

		 cell.foo = 1;
		 cell.thickness = 1;
		 cell.outflow[0] = 1;
		 cell.outflow[1] = 2;
		 cell.outflow[2] = 3;
		 cell.outflow[3] = 4;

		 return &cell;
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
	//Populates a max of 4 neighbours
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
	

	//int represents the pointer value, I have no idea how to get CUDA specific information as to what datatype to store a pointer in....can't use void* as
	//I can't make it modifible...TODO look into this later.
	__device__ __host__ void getVonNeumannNeighbourhood2(int* neighbours, void* g_data, int x, int y, int xDIM,size_t structSize) {
		
			
		void* myPointer = (void*)(&g_data + 1);


		neighbours[0] = (int)myPointer;
		(void*)(&g_data + structSize);


	// [0,-1]
		if ( y != 0)
			neighbours[0] = (int)(&g_data + ((x + y - 1) * structSize));

		// [-1,0]
		if (x != 0)
			neighbours[1] = (int)(&g_data + ((x - xDIM + y) * structSize));

		// [1,0]
		if (x != xDIM - 1)
			neighbours[2] =  (int)(&g_data + ((x + xDIM + y) * structSize));

		// [0,1]
		if (y != xDIM -1 )
			neighbours[3] = (int)(&g_data + ((x + y + 1) * structSize));
	}
	//Populates a max of 4 neighbours
	__device__ __host__ void getVonNeumannNeighbourhood3(int* neighbours, int x, int y, int xDIM) {
		
		bool xBounds = (x / xDIM) < xDIM -1 ;

		// [0,-1]
		if ( y != 0){
			neighbours[0] = x + y - 1;
		}

		// [-1,0]
		if (x != 0){
			neighbours[1] = x - xDIM + y;
		}

		// [1,0]
		if (xBounds){
			neighbours[2] = x + xDIM + y;
		}

		// [0,1]
		if (y != xDIM - 1) {
			neighbours[3] = x + y + 1;
		}
	}
};

