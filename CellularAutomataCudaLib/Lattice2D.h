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
* This class representats a Two Dimensional lattice. It supports neighbourhood information for both a Moore neighbourhood and a Von Neumann.
* More information on these can be found at.
* Moore : http://en.wikipedia.org/wiki/Moore_neighborhood
* Von Neumann : http://en.wikipedia.org/wiki/Von_Neumann_neighborhood
* 
*/
class Lattice2D : public AbstractLattice
{

public:
	/**
	* Default constructor. No size dimensions will be set.
	*/
	DLLExport Lattice2D(void);

	/**
	* Sets up the lattice with the specific grid along with the given sizes. The grid can consist of any type or stuct. It must
	* be a one dimensional version of a 2D lattice.
	*@param grid The new Cell grid to get assigned to the Lattice. This must be a one dimensional array.
	*@param sizeX The x size of the grid as 2D.
	*@param sizeY The y size of the grid as 2D.
	*/
	DLLExport Lattice2D(void* grid, int sizeX, int sizeY);

	/**
	* Sets up the lattice with integer cells that contain random 0-1 data. 
	* A seed value of 1 will lead to 1/1 cells having a state of 1 (100%).
	* A seed value of 2 will lead to 1/2 cells having a state of 1 (50%).
	* A seed value of 3 will lead to 1/3 cells having a state of 1 (33%).
	* ...etc.
	*@param sizeX The x size of the grid as 2D.
	*@param sizeY The y size of the grid as 2D.
	*@param seed The seed value to use to set the random data.
	*/
	DLLExport Lattice2D(int sizeX,int sizeY,int seed); //random data

	/**
	* A default destructor.
	*/ 
	DLLExport ~Lattice2D(void);

	
	unsigned int yDIM; /**< The y size of the 2D lattice. This variable is only public for __device__ access, hosts should use the getters and setters.*/
	
	/**
	* To allocate the appropriate amount of memory on the GPU. The lattice class must specify how big it is. This is the actual class itself.
	* ie. a typical implemented would be.
	* return sizeof(LatticeXD);
	*@return The size of the clas in bytes.
	*/
	__host__ virtual size_t size() const { return sizeof(Lattice2D); }

	//Always better to use constants than defines (effective C++)
	//Using 8 and 4 as the number of neighbours..
	static const int MOORE = 8;  /**< The Moore neighbourhood*/
	static const int VON_NEUMANN = 4;  /**< The Von Neumann neighbourhood*/

	/**
	* Provides the location of the cell that surround the center cell based upon the neighbourhood type.
	* The outNeighbourStates array is passed to this function and subsequently populated with the available neighbourhood cell locations.
	* This allows the rule class to use this information as they see fit. For example, in the generations case, the neighbourhood cells are
	* totaled up to decide if the cell dies/survives. 
	*@param outNeighbourStates An array that should be set to -1. It must have a maximum size to accommodate the potential max neighbours, in this case 8. The array contains neighbours location in the one-d lattice grid.
	*@param x The x co-ordinate of our center cell in 2D space.
	*@param y The x co-ordinate of our center cell in 2D space.
	*@param xSize The total size of the X dimension.
	*@param ySize the total size of the Y dimension
	*/
	__device__ __host__ void getNeighbourhood(int* outNeighbourStates, int x, int y, int xSize, int ySize) {

		switch(neighbourhoodType) {
		case MOORE:
			getMooresNeighbourhood(outNeighbourStates,x,y,xSize,ySize);
			break;
		case VON_NEUMANN:
			getVonNeumannNeighbourhood(outNeighbourStates,x,y,xSize,ySize);
			break;
		default:
			break;
		}
	}

private:
	//Populates a max of 8 neighbours
	__device__ __host__ void getMooresNeighbourhood(int* neighbours,int x, int y, int xDIM, int yDIM) {
		
		x = x * xDIM;

		bool xBounds = (x / xDIM) < xDIM -1 ;

		// [-1,-1]
		if (x != 0 && y != 0)
			neighbours[0] = x - yDIM + y - 1;

		// [0,-1]
		if ( y != 0)
			neighbours[1] = x + y - 1;

		// [1,-1]
		if (xBounds && y != 0 )
			neighbours[2] = x + yDIM + y - 1;

		// [-1,0]
		if (x != 0)
			neighbours[3] = x - yDIM + y;

		// [1,0]
		if (xBounds)
			neighbours[4] = x + yDIM + y;

		// [-1,1]
		if (x != 0 && y != yDIM - 1)
			neighbours[5] = x - yDIM + y + 1;

		// [0,1]
		if (y != yDIM -1 )
			neighbours[6] = x + y + 1;

		// [1,1]
		if (xBounds && y != yDIM - 1 )
			neighbours[7] = x + yDIM + y + 1;

	}

	//Populates a max of 4 neighbours
	__device__ __host__ void getVonNeumannNeighbourhood(int* neighbours, int x, int y, int xDIM, int yDIM) {
		
		x = x * xDIM;

		bool xBounds = (x / yDIM) < xDIM -1 ;

		// [0,-1]
		if ( y != 0){
			neighbours[0] = x + y - 1;
		}

		// [-1,0]
		if (x != 0){
			neighbours[1] = x - yDIM + y;
		}

		// [1,0]
		if (xBounds){
			neighbours[2] = x + yDIM + y;
		}

		// [0,1]
		if (y != yDIM - 1) {
			neighbours[3] = x + y + 1;
		}
	}
};


