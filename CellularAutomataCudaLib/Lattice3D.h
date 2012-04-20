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
/**
* This class representats a Three Dimensional lattice. It supports neighbourhood information for both a Moore neighbourhood and a Von Neumann.
* More information on these can be found at.
* Moore : http://en.wikipedia.org/wiki/Moore_neighborhood
* Von Neumann : http://en.wikipedia.org/wiki/Von_Neumann_neighborhood
* 
* The 3D versions extend their two dimensional definitions. Moore has 26 neighbours, Von Neumann has 6.
*/
class Lattice3D : public AbstractLattice
{
public:
	/**
	* Default constructor. No size dimensions will be set.
	*/
	DLLExport Lattice3D(void);

	/**
	* Sets up the lattice with the specific grid along with the given sizes. The grid can consist of any type or stuct. It must
	* be a one dimensional version of a 3D lattice.
	*@param grid The new Cell grid to get assigned to the Lattice. This must be a one dimensional array.
	*@param sizeX The x size of the grid as 3D.
	*@param sizeY The y size of the grid as 3D.
	*@param sizeZ The z size of the grid as 3D.
	*/
	DLLExport Lattice3D(void* grid, int sizeX, int sizeY, int sizeZ);

	/**
	* Sets up the lattice with integer cells that contain random 0-1 data. 
	* A seed value of 1 will lead to 1/1 cells having a state of 1 (100%).
	* A seed value of 2 will lead to 1/2 cells having a state of 1 (50%).
	* A seed value of 3 will lead to 1/3 cells having a state of 1 (33%).
	* ...etc.
	*@param sizeX The x size of the grid as 3D.
	*@param sizeY The y size of the grid as 3D.
	*@param sizeZ The z size of the grid as 3D.
	*@param seed The seed value to use to set the random data.
	*/
	DLLExport Lattice3D(int sizeX,int sizeY,int sizeZ,int seed); //random data

	/**
	* A default destructor.
	*/ 
	DLLExport virtual ~Lattice3D(void);

	/**
	* To allocate the appropriate amount of memory on the GPU. The lattice class must specify how big it is. This is the actual class itself.
	* ie. a typical implemented would be.
	* return sizeof(LatticeXD);
	*@return The size of the clas in bytes.
	*/
	__host__ virtual size_t size() const { return sizeof(Lattice3D); }

	unsigned int yDIM; /**< The y size of the 3D lattice. This variable is only public for __device__ access, hosts should use the getters and setters.*/
	unsigned int zDIM;/**< The z size of the 3D lattice. This variable is only public for __device__ access, hosts should use the getters and setters.*/

	unsigned int *neighbourCount; /**< Optional parameter that will tally how many neighbours a cell has. This can be used by clients to optimse rendering. */

	//Always better to use constants than defines (effective C++)
	static const int MOORE_3D = 26; /**< The Moore neighbourhood*/
	static const int VON_NEUMANN_3D = 6; /**< The Von Neumann neighbourhood*/

	/**
	* Provides the location of the cell that surround the center cell based upon the neighbourhood type.
	* The outNeighbourStates array is passed to this function and subsequently populated with the available neighbourhood cell locations.
	* This allows the rule class to use this information as they see fit. For example, in the generations case, the neighbourhood cells are
	* totaled up to decide if the cell dies/survives. 
	*@param outNeighbourStates An array that should be set to -1. It must have a maximum size to accommodate the potential max neighbours, in this case 8. The array contains neighbours location in the one-d lattice grid.
	*@param x The x co-ordinate of our center cell in 3D space.
	*@param y The x co-ordinate of our center cell in 3D space.
	*@param z The z co-ordinate of our center cell in 3D space.
	*@param DIM The total size of the X dimension.
	*/
	__device__ __host__ void getNeighbourhood(int* outNeighbourStates, int x, int y, int z, int DIM){

		switch(neighbourhoodType) {
		case MOORE_3D:
			get3dMooresNeighbourhood(outNeighbourStates,x,y,z, DIM);
			break;
		case VON_NEUMANN_3D:
			get3dVonNeumannNeighbourhood(outNeighbourStates,x,y,z,DIM);
			break;
		default:
			break;
		}
	}

private:
	//probably a much better way to figure out the moores neighbourhood
	__device__ __host__ void get3dMooresNeighbourhood(int* neighbourStates, int x, int y, int z, int DIM) {

		x = x * DIM;
		z = z * DIM * DIM;

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

		x = x * xDIM;
		z = z * xDIM * xDIM;


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