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

class Abstract3DCA : public AbstractLattice
{
public:
	DLLExport Abstract3DCA(void);
	DLLExport Abstract3DCA(int,int); //random data
	DLLExport Abstract3DCA(unsigned int*, int);

	DLLExport virtual ~Abstract3DCA(void); //Force use of derived constructor 
	
	__host__ virtual size_t size() const { return sizeof(this); }
	
	unsigned int *neighbourCount;

	//Always better to use constants than defines (effective C++)
	static const int MOORE_3D = 26;
	static const int VON_NEUMANN_3D = 6;
		
	__device__ __host__ void getNeighbourhood(int* neighbourStates, unsigned int* g_data, int gLocation){

		int z = gLocation / (DIM * DIM);
		int x = (gLocation - (z * DIM * DIM)) / DIM;
		int y = gLocation % DIM;

		int zAltered = z * DIM * DIM;
		int xAltered = x * DIM;

		switch(neighbourhoodType) {
		case MOORE_3D:
			get3dMooresNeighbourhood(neighbourStates,g_data,xAltered,y,zAltered,DIM);
			break;
		case VON_NEUMANN_3D:
			get3dVonNeumannNeighbourhood(neighbourStates,g_data,xAltered,y,zAltered,DIM);
			break;
		default:
			break;
		}
	}

__device__ __host__ void getNeighbourhood(int* neighbourStates, int x, int y, int z,unsigned int* g_data, int gLocation){

		switch(neighbourhoodType) {
		case MOORE_3D:
			get3dMooresNeighbourhood(neighbourStates,g_data,x,y,z,DIM);
			break;
		case VON_NEUMANN_3D:
			get3dVonNeumannNeighbourhood(neighbourStates,g_data,x,y,z,DIM);
			break;
		default:
			break;
		}
	}

	//probably a much better way to figure out the moores neighbourhood
	__device__ __host__ void get3dMooresNeighbourhood(int* neighbourStates, unsigned int* g_data, int x, int y, int z, int xDIM) {
		int zDIM = xDIM * xDIM;

		//bool xBounds = (x + (xDIM - 1)/xDIM) < xDIM;
		//bool zBounds = (z + (zDIM - 1)/zDIM) < zDIM;

		bool xBounds = (x / xDIM) < xDIM -1;
		bool zBounds = (z / zDIM) < xDIM -1;

		// [-1,-1,-1]
		if (x != 0 && y != 0 && z != 0)
			neighbourStates[0] = g_data[x - xDIM + y - 1 + z - zDIM];

		// [-1,-1,0]
		if (x != 0 && y != 0)
			neighbourStates[1] = g_data[x - xDIM + y - 1 + z];

		// [-1,-1,1]
		if (x != 0 && y != 0 && zBounds)
			neighbourStates[2] = g_data[x - xDIM + y - 1 + z + zDIM ];


		// [-1,0,-1]
		if (x != 0 && z != 0)
			neighbourStates[3] = g_data[x - xDIM + y + z - zDIM];

		// [-1,0,0]
		if (x != 0)
			neighbourStates[4] = g_data[x - xDIM + y  + z];
				
		// [-1,0,1]
		if (x != 0  && zBounds)
			neighbourStates[5] = g_data[x - xDIM + y + z + zDIM ];
		
		// [-1,1,-1]
		if (x != 0 && y != xDIM -1 && z != 0 )
			neighbourStates[6] = g_data[x - xDIM + y + 1 + z - zDIM];
				
		// [-1,1,0]
		if (x != 0 && y != xDIM -1 )
			neighbourStates[7] = g_data[x - xDIM + y + 1  + z];

		// [-1,1,1
		if (x != 0 && y != xDIM -1 && zBounds)
			neighbourStates[8] = g_data[x - xDIM + y + 1 + z + zDIM ];
				
		//x = 0

		// [0,-1,-1]
		if ( y != 0 && z != 0)
			neighbourStates[9] = g_data[x + y - 1 + z - zDIM];

		// [0,-1,0]
		if ( y != 0)
			neighbourStates[10] = g_data[x + y - 1  + z];

		// [0,-1,1]
		if ( y != 0  && zBounds)
			neighbourStates[11] = g_data[x + y - 1 + z + zDIM];
					
		// [0,0,-1]
		if (z != 0)
			neighbourStates[12] = g_data[x + y + z - zDIM];
					
		//0,0,0

		// [0,0,1]
		if (zBounds)
			neighbourStates[13] = g_data[x + y + z + zDIM ];
				
		// [0,1,-1]
		if (y != xDIM -1 && z != 0 )
			neighbourStates[14] = g_data[x + y + 1 + z - zDIM];
				
		// [0,1,0]
		if (y != xDIM -1 )
			neighbourStates[15] = g_data[x + y + 1  + z];
			
		// [0,1,1]
		if (y != xDIM -1 && zBounds )
			neighbourStates[16] = g_data[x + y + 1 + z + zDIM ];
				
		//x = 1

		// [1,-1,-1]
		if (xBounds && y != 0 && z != 0)
			neighbourStates[17] = g_data[x + xDIM + y - 1 + z - zDIM];
				
		// [1,-1,0]
		if (xBounds && y != 0 )
			neighbourStates[18] = g_data[x + xDIM + y - 1  + z];
				
		// [1,-1,1]
		if (xBounds && y != 0  && zBounds)
			neighbourStates[19] = g_data[x + xDIM + y - 1 + z + zDIM ];
				
		// [1,0,-1]
		if (xBounds && z != 0)
			neighbourStates[20] = g_data[x + xDIM + y + z - zDIM];
				
		// [1,0,0]
		if (xBounds)
			neighbourStates[21] = g_data[x + xDIM + y  + z];
				
		// [1,0,1]
		if (xBounds  && zBounds)
			neighbourStates[22] = g_data[x + xDIM + y + z + zDIM ];


		// [1,1,-1]
		if (xBounds && y != xDIM - 1 && z != 0 )
			neighbourStates[23] = g_data[x + xDIM + y + 1 + z - zDIM];

		// [1,1,0]
		if (xBounds && y != xDIM - 1 )
			neighbourStates[24] = g_data[x + xDIM + y + 1  + z];

		// [1,1,1]
		if (xBounds && y != xDIM - 1 && zBounds )
			neighbourStates[25] = g_data[x + xDIM + y + 1 + z + zDIM ];
	}


		//probably a much better way to figure out the moores neighbourhood
	__device__ __host__ void get3dVonNeumannNeighbourhood(int* neighbourStates, unsigned int* g_data, int x, int y, int z, int xDIM) {
		int zDIM = xDIM * xDIM;

		//bool xBounds = (x + (xDIM - 1)/xDIM) < xDIM;
		//bool zBounds = (z + (zDIM - 1)/zDIM) < zDIM;

		bool xBounds = (x / xDIM) < xDIM -1;
		bool zBounds = (z / zDIM) < xDIM -1;

		//x = -1

		// [-1,0,0]
		if (x != 0)
			neighbourStates[0] = g_data[x - xDIM + y  + z];
				
		//x = 0
		// [0,-1,0]
		if ( y != 0)
			neighbourStates[1] = g_data[x + y - 1  + z];

		// [0,0,-1]
		if (z != 0)
			neighbourStates[2] = g_data[x + y + z - zDIM];
		
		// [0,0,1]
		if (zBounds)
			neighbourStates[3] = g_data[x + y + z + zDIM ];
				
		// [0,1,0]
		if (y != xDIM -1 )
			neighbourStates[4] = g_data[x + y + 1  + z];
				
		//x = 1
		// [1,0,0]
		if (xBounds)
			neighbourStates[5] = g_data[x + xDIM + y  + z];

	}

};