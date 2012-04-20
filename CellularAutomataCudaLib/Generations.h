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

#ifndef GENERATIONS_H
#define GENERATIONS_H

#include "device_launch_parameters.h"
#include "AbstractCellularAutomata.h"
#include "Lattice2D.h"
#include "cuda.h"
#include "Totalistic.h"
#include <map>
#include <vector>

using namespace std;

/**
* The "Generations" game rules are very close to those from Life, with one addition: the cells' history.
* Cells that would simply die in "Life" are only getting older in "Generations". They cannot give birth to new cells, 
* but they occupy the space of the lattice, thus changing the rules radically.
*
* The generations game is a totalistic game, in that it sums up the values of neighbours around it to determine it's next state.
* This implementation is 2 dimensional.
*/
class Generations : public Totalistic{

public :
	DLLExport __device__ __host__ Generations();

	DLLExport __device__ __host__ ~Generations();


	Lattice2D *lattice; /**< The 2D lattice used by the Rule. This is only public for access by Kernel. Host could should use the getter/setter provided */

public:

	/**
	* Gets the currently set latice.
	* Note : this method is defined inline due to being accessed by the GPU.
	*@return The current lattice
	*/
	__host__ __device__ Lattice2D* getLattice() {
		return lattice;
	}


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
	__host__ __device__  int applyFunction(void* g_data, int x, int y, int xDIM, int yDIM) { 

		int gridLoc = x * yDIM + y;

		unsigned int* cellData = (unsigned int*)g_data;

		int state = cellData[gridLoc];

		int newState = state;

		int temp = 0;
		//generations specialism
		if (state > 1) {
			if(state < noStates - 1) {
				temp = state + 1;
				newState = setNewState(lattice,temp,state);			
			}
		}
		else {
			//This is signed so we can default to -1, this shows NO neighbour, a result of 0 means a neighbour who's state is zero
			//This may cause complications as states are stored in unsigned ints...
			int neighbourhoodStates[8];

			//set as -1 by default.
			for(int i = 0; i < 8; i++) {
				neighbourhoodStates[i] = -1; 
			}

			//Populate neighbours states.
			lattice->getNeighbourhood(neighbourhoodStates,x,y,xDIM,yDIM);




			//Should probably move this code to TOTALISTIC CLASS, but it might slow it down..
			int liveCells = 0;//getLiveCellCount(neighbourhoodStates,lattice->maxBits,lattice->neighbourhoodType);

			for(int i = 0; i < lattice->getNeighbourhoodType(); ++i) {
				if(cellData[neighbourhoodStates[i]] != -1) {

			/*		int neighX = neighbourhoodStates[i] / xDIM;
					int neighY = neighbourhoodStates[i] % xDIM;
					
					neighY = neighY - (blockIdx.y * blockDim.y);
					neighX = neighX - (blockIdx.x * blockDim.x);*/

					//And back to our flat shared array..
					if((cellData[neighbourhoodStates[i]] & lattice->getMaxBits()) == 1) //This cell's state is alive.
						++liveCells;
				}
			}


			//we only care about neighbours when we know we're in a ready state
			//int liveCells = Totalistic::getLiveCellCount(neighbourhoodStates,lattice->maxBits,lattice->neighbourhoodType);

			if(state == 1) {

				for (int i = 0; i < surviveSize; i++) {
					if (liveCells == surviveNo[i]) newState = setNewState(lattice,1,state);
				}

			}
			else if(state == 0) {

				for (int i = 0; i < bornSize; ++i) {		
					if (liveCells == bornNo[i]) newState = setNewState(lattice,1,state);
				}

			}

			if (state == 1 && newState == state) {
				if (state < noStates - 1) { //This guards against 2 state generations
					newState = setNewState(lattice,2,state);
				}
			}
		}

		//Potential bug here, could writing corrupt our data ??
		cellData[gridLoc] = newState;

		return 0;
	}
};


#endif