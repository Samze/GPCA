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

class Lattice3D : public AbstractLattice
{
public:
	DLLExport Lattice3D(void);
	DLLExport Lattice3D(int xDIM,int yDIM,int zDIM,int); //random data
	DLLExport Lattice3D(void*, int xSize, int ySize, int zSize);

	DLLExport virtual ~Lattice3D(void); //Force use of derived constructor 
	
	__host__ virtual size_t size() const { return sizeof(this); }
	
	unsigned int yDIM;
	unsigned int zDIM;

	unsigned int *neighbourCount;

	//Always better to use constants than defines (effective C++)
	static const int MOORE_3D = 26;
	static const int VON_NEUMANN_3D = 6;


__device__ __host__ void getNeighbourhood(int* neighbourStates, int x, int y, int z, int DIM){

		switch(neighbourhoodType) {
		case MOORE_3D:
			get3dMooresNeighbourhood(neighbourStates,x,y,z, DIM);
			break;
		case VON_NEUMANN_3D:
			get3dVonNeumannNeighbourhood(neighbourStates,x,y,z,DIM);
			break;
		default:
			break;
		}
	}
	
	//probably a much better way to figure out the moores neighbourhood
	__device__ __host__ void get3dMooresNeighbourhood(int* neighbourStates, int x, int y, int z, int DIM) {
		
		int zDIM = DIM * DIM;
		
		bool xBounds = (x / DIM) < DIM -1;
		bool zBounds = (z / zDIM) < DIM -1;

		// [-1,-1,-1]
		if (x != 0 && y != 0 && z != 0)
			neighbourStates[0] = x - DIM + y - 1 + z - zDIM;

		// [-1,-1,0]
		if (x != 0 && y != 0)
			neighbourStates[1] = x - DIM + y - 1 + z;

		// [-1,-1,1]
		if (x != 0 && y != 0 && zBounds)
			neighbourStates[2] = x - DIM + y - 1 + z + zDIM;


		// [-1,0,-1]
		if (x != 0 && z != 0)
			neighbourStates[3] = x - DIM + y + z - zDIM;

		// [-1,0,0]
		if (x != 0)
			neighbourStates[4] = x - DIM + y  + z;
				
		// [-1,0,1]
		if (x != 0  && zBounds)
			neighbourStates[5] = x - DIM + y + z + zDIM;
		
		// [-1,1,-1]
		if (x != 0 && y != DIM -1 && z != 0 )
			neighbourStates[6] = x - DIM + y + 1 + z - zDIM;
				
		// [-1,1,0]
		if (x != 0 && y != DIM -1 )
			neighbourStates[7] = x - DIM + y + 1  + z;

		// [-1,1,1
		if (x != 0 && y != DIM -1 && zBounds)
			neighbourStates[8] = x - DIM + y + 1 + z + zDIM;
				
		//x = 0

		// [0,-1,-1]
		if ( y != 0 && z != 0)
			neighbourStates[9] = x + y - 1 + z - zDIM;

		// [0,-1,0]
		if ( y != 0)
			neighbourStates[10] = x + y - 1  + z;

		// [0,-1,1]
		if ( y != 0  && zBounds)
			neighbourStates[11] = x + y - 1 + z + zDIM;
					
		// [0,0,-1]
		if (z != 0)
			neighbourStates[12] = x + y + z - zDIM;
					
		//0,0,0

		// [0,0,1]
		if (zBounds)
			neighbourStates[13] = x + y + z + zDIM;
				
		// [0,1,-1]
		if (y != DIM -1 && z != 0 )
			neighbourStates[14] = x + y + 1 + z - zDIM;
				
		// [0,1,0]
		if (y != DIM -1 )
			neighbourStates[15] = x + y + 1  + z;
			
		// [0,1,1]
		if (y != DIM -1 && zBounds )
			neighbourStates[16] = x + y + 1 + z + zDIM;
				
		//x = 1

		// [1,-1,-1]
		if (xBounds && y != 0 && z != 0)
			neighbourStates[17] = x + DIM + y - 1 + z - zDIM;
				
		// [1,-1,0]
		if (xBounds && y != 0 )
			neighbourStates[18] = x + DIM + y - 1  + z;
				
		// [1,-1,1]
		if (xBounds && y != 0  && zBounds)
			neighbourStates[19] = x + DIM + y - 1 + z + zDIM;
				
		// [1,0,-1]
		if (xBounds && z != 0)
			neighbourStates[20] = x + DIM + y + z - zDIM;
				
		// [1,0,0]
		if (xBounds)
			neighbourStates[21] = x + DIM + y  + z;
				
		// [1,0,1]
		if (xBounds  && zBounds)
			neighbourStates[22] = x + DIM + y + z + zDIM;


		// [1,1,-1]
		if (xBounds && y != DIM - 1 && z != 0 )
			neighbourStates[23] = x + DIM + y + 1 + z - zDIM;

		// [1,1,0]
		if (xBounds && y != DIM - 1 )
			neighbourStates[24] = x + DIM + y + 1  + z;

		// [1,1,1]
		if (xBounds && y != DIM - 1 && zBounds )
			neighbourStates[25] = x + DIM + y + 1 + z + zDIM;
	}

	
		//probably a much better way to figure out the moores neighbourhood
	__device__ __host__ void get3dVonNeumannNeighbourhood(int* neighbourStates,int x, int y, int z, int DIM) {
		
		int zDIM = DIM * DIM;

		bool xBounds = (x / DIM) < DIM -1;
		bool zBounds = (z / zDIM) < DIM -1;

		//x = -1

		// [-1,0,0]
		if (x != 0)
			neighbourStates[0] = x - DIM + y  + z;
				
		//x = 0
		// [0,-1,0]
		if ( y != 0)
			neighbourStates[1] = x + y - 1  + z;

		// [0,0,-1]
		if (z != 0)
			neighbourStates[2] = x + y + z - zDIM;
		
		// [0,0,1]
		if (zBounds)
			neighbourStates[3] = x + y + z + zDIM;
				
		// [0,1,0]
		if (y != DIM -1 )
			neighbourStates[4] = x + y + 1  + z;
				
		//x = 1
		// [1,0,0]
		if (xBounds)
			neighbourStates[5] = x + DIM + y  + z;

	}

};