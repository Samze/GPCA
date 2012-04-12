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

#ifndef OUTERTOTALISTIC3D_H
#define OUTERTOTALISTIC3D_H

#include "device_launch_parameters.h"
#include "AbstractCellularAutomata.h"
#include "Totalistic.h"
#include "Lattice3D.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include <map>
#include <vector>

using namespace std;

/**
* The "Outer Totalisitc" game rules combine two facets. Firstly it is a totalistic game, in that it sums up the values 
* of neighbours around it to determine it's next state. Secondly it is Outer Totalistic, so a cell also takes into account
* it's own state as well as that of it's neighbours. 
*
* The famous game of life is an example of a Outer totalisitic CA.
* This implementation is 3 dimensional.
*/
class OuterTotalistic3D : public Totalistic {

public :
	DLLExport __device__ __host__ OuterTotalistic3D();
	DLLExport __device__ __host__ ~OuterTotalistic3D();

	Lattice3D *lattice; /**< The 2D lattice used by the Rule. This is only public for access by Kernel. Host could should use the getter/setter provided */


public:
	__host__ virtual void setLattice(AbstractLattice* newLattice);

	/**
	* Return a list of dynamic pointers to be put onto the GPU.
	* This is used to dynamically allocate data on the GPU. This will be used for lattice information. But also
	* Used for survial/born state data.
	*@return A map containing the address of the pointer that holds the data, along with it's size. 
	*/
	__host__ map<void**, size_t>* getDynamicArrays();

	/**
	* Gets the currently set latice.
	* Note : this method is defined inline due to being accessed by the GPU.
	*@return The current lattice
	*/
	__host__ __device__ Lattice3D* getLattice() { 
		return lattice;
	}

	/**
	* This is the transition function for a single cell. The pointers are passed here instead of being assumed as
	* they can contained shared data pointers instead of global pointers. This is a special type of GPGPU memory that
	* has significant performance benefits.
	* Note : this method is defined inline due to being accessed by the GPU.
	* @param g_data A pointer which contains a flat array of lattice state data.
	* @param x The x co-ordinate of the cell to apply the function to.
	* @param y The y co-ordinate of the cell to apply the function to.
	* @param z The y co-ordinate of the cell to apply the function to.
	* @param xDIM The total x size
	*/
	__host__ __device__ int applyFunction(void* g_data, int x, int y, int z, int xDIM) { 

		int xAltered = x * xDIM;
		int zAltered = z * xDIM * xDIM;

		unsigned int* cellData = (unsigned int*)g_data;
		
		int gridLoc = xAltered + zAltered + y;

		int state = cellData[gridLoc];
	//	//We want know about neighbours even if we're not using them to set the next state, this is 
	//	//so they can not be rendered by the viewer. To speed up the processing, move this to the else

		int neighbourhoodStates[26];//= {-1};
	//
	//	//set as -1 by default.
		//TODO Add unrolling here!
		for(int i = 0; i < 26; i++) {
			neighbourhoodStates[i] = -1; 
		}


	//	//Populate neighbours states.
		lattice->getNeighbourhood(neighbourhoodStates,x,y,z,xDIM);


		//populate our neighbours with their cell state values 
		//TODO Add unrolling here!
		for(int i= 0; i <26; i++) {
			if(neighbourhoodStates[i] != -1) {
				neighbourhoodStates[i] = cellData[neighbourhoodStates[i]];
			}
		}

		unsigned int liveCells = Totalistic::getLiveCellCount(neighbourhoodStates,lattice->getMaxBits(),lattice->getNeighbourhoodType());

	//	//int liveCells = getNeighbourhood(g_data, xAltered, y, zAltered, xDIM);

	//	//This is for culling of cubes surrounded on all sides.
		
		//lattice->neighbourCount[xAltered + y + zAltered] = liveCells;
		

		unsigned int newState = state;

		for (int i = 0; i < surviveSize; i++) {
			if (state == 1 && liveCells == surviveNo[i]) newState = setNewState(lattice,1,state);
		}

		for (int i = 0; i < bornSize; ++i) {		
			if (state == 0 && liveCells == bornNo[i]) newState = setNewState(lattice,1,state);
		}
		
		//Potential bug here, could writing corrupt our data ??
		cellData[gridLoc] = newState;
		return 0;
	}
};


#endif
