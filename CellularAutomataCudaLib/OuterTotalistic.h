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

#ifndef OUTER_TOTALISTIC_H_
#define OUTER_TOTALISTIC_H_

#include "device_launch_parameters.h"
#include "AbstractCellularAutomata.h"
#include "Lattice2D.h"
#include "Totalistic.h"
#include "cuda.h"
#include <map>
#include <vector>

using namespace std;

/**
* The "Outer Totalisitc" game rules combine two facets. Firstly it is a totalistic game, in that it sums up the values 
* of neighbours around it to determine it's next state. Secondly it is Outer Totalistic, so a cell also takes into account
* it's own state as well as that of it's neighbours. 
*
* The famous game of life is an example of a Outer totalisitic CA.
* This implementation is 2 dimensional.
*/
class OuterTotalistic : public Totalistic {

public :
	DLLExport __device__ __host__ OuterTotalistic();
	DLLExport __device__ __host__ ~OuterTotalistic();


	//This is only public for access by Kernel...
	Lattice2D *lattice; /**< The 2D lattice used by the Rule. This is only public for access by Kernel. Host could should use the getter/setter provided */

public:

	/**
	* Return a list of dynamic pointers to be put onto the GPU.
	* This is used to dynamically allocate data on the GPU. This will be used for lattice information. But also
	* Used for survial/born state data.
	*@return A map containing the address of the pointer that holds the data, along with it's size. 
	*/
	__host__ map<void**, size_t>* getDynamicArrays();

	/**
	* Sets this Rules lattice.
	*@param newLattice The new Lattice to set
	*/
	__host__ virtual void setLattice(AbstractLattice* newLattice);
	
	/**
	* Gets the currently set latice.
	* Note : this method is defined inline due to being accessed by the GPU.
	*@return The current lattice
	*/
	__host__ __device__ Lattice2D* getLattice() { 
		return lattice;
	}

	/**
	* This is the transition function for a single cell. The pointers are passed here instead of being assumed as
	* they can contained shared data pointers instead of global pointers. This is a special type of GPGPU memory that
	* has significant performance benefits.
	* Note : this method is defined inline due to being accessed by the GPU.
	* @param g_data A pointer which contains a flat array of lattice state data.
	* @param x The x co-ordinate of the cell to apply the function to
	* @param y The y co-ordinate of the cell to apply the function to
	* @param xDIM The total x size
	* @param yDIM The total y size
	*/
	__device__ __host__ int applyFunction(void* g_data, int x, int y, int xDIM,int yDIM) { 
		
		int gridLoc = x * yDIM + y;

		unsigned int* cellData = (unsigned int*)g_data;

		int state = cellData[gridLoc];

		int newState = state;
		
		int neighbourhoodStates[8];
	
		//set as -1 by default.
		for(int i = 0; i < 8; i++) {
			neighbourhoodStates[i] = -1; 
		}

		lattice->getNeighbourhood(neighbourhoodStates,x,y,xDIM,yDIM);

		//Should probably move this code to TOTALISTIC CLASS, but it might slow it down..
		int liveCells = 0;//getLiveCellCount(neighbourhoodStates,lattice->maxBits,lattice->neighbourhoodType);

		for(int i = 0; i < lattice->getNeighbourhoodType(); ++i) {
			if(cellData[neighbourhoodStates[i]] != -1) {
				if((cellData[neighbourhoodStates[i]] & lattice->getMaxBits()) == 1) //This cell's state is alive.
					++liveCells;
			}
		}

		for (int i = 0; i < surviveSize; i++) {
			if (state == 1 && liveCells == surviveNo[i]) newState = setNewState(lattice,1,state);
		}

		for (int i = 0; i < bornSize; ++i) {		
			if (state == 0 && liveCells == bornNo[i]) newState = setNewState(lattice,1,state);
		}

		cellData[gridLoc] = newState;
		
		return 0;
	}
};


#endif
