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
#include "cuda.h"

/**
* This class represents a Cellular Automata that is Totalistic. A totalistic CA cell counts the states of it's neighbours
* to decide what it's state will be at time + 1. This class holds survival and born information for CAs.
*
* Survive information - How many cells around the central cell must be active for this cell to survive.
* Born information - How many cells around the central cell must be active for this cell to be born.
*
* State data is packed into bits as follows.
* NEW_STATE_BITS|OLD_STATE_BITS
*
* For example if our states can be 0 or 1, dead or alive, then we would have two bits to represent this data.
* The first for the current state, the second for the new state.
*
* 0 1 - Currently alive, next state is dead.
* 1 1 - Currently alive, next state alive.
* 0 1 - Currently dead, next state alive.
*
* This bitpacking can be used for up to 2147483647 state for 32bit ints.
*/
class Totalistic : public AbstractCellularAutomata{

public:
	DLLExport __device__  __host__ Totalistic();

	/**
	* Virtual destructor. This is an abstract class.
	*/
	DLLExport __device__  __host__ virtual ~Totalistic();
	 
	/**
	* Sets the number of states
	*@param states The new number of states.
	*/
	DLLExport __host__ virtual void setStates(unsigned int states);
	
	/**
	* Returns the number of states
	*@return The number of states
	*/
	DLLExport __host__ int getNoStates() { return noStates;}

	/*
	* Sets the survival numbers for this rule. This will be a pointer to a list of integers.
	* For example an array of [2,3,4]. Means that any cell with 2,3 or 4 neighbours will survive.
	*@param list A pointer to the integer list of survival states
	*@param size The number of survival states
	*/
	__host__ void setSurviveNo(int* list, int size) {
		surviveNo = list;
		surviveSize = size;
	}

	/*
	* Sets the born numbers for this rule. This will be a pointer to a list of integers.
	* For example an array of [2,3,4]. Means that any cell with 2,3 or 4 neighbours will be born.
	*@param list A pointer to the integer list of born states.
	*@param size The number of born states.
	*/
	__host__ void setBornNo(int* list, int size) {
		bornNo = list;
		bornSize = size;
	}

	int* surviveNo; /**< Survival numbers for the rule. This is only public for access by Kernel. Host could should use the getter/setter provided */
    int  surviveSize;  /**< Survival numbers size. This is only public for access by Kernel. Host could should use the getter/setter provided */
	 
	int* bornNo; /**< Born numbers for the rule. This is only public for access by Kernel. Host could should use the getter/setter provided */
    int bornSize;  /**< Born numbers size. This is only public for access by Kernel. Host could should use the getter/setter provided */

	
	/*
	* Gets the number of live cells for a set of neighbour states. It determines this by checking
	*
	*@param neighbourhoodStates The 
	*@param maxBits
	*@param neighbourType
	*@return Returns the number of live cells around the cell.
	*/
	__device__ __host__  static unsigned int getLiveCellCount(int* neighbourhoodStates, int maxBits, int neighbourType) {

		unsigned int numLiveCells =0;

		for(int i = 0; i < neighbourType; ++i) {
			if(neighbourhoodStates[i] != -1) 
				if((neighbourhoodStates[i] & maxBits) == 1) //This cell's state is alive.
					++numLiveCells;
		}

		return numLiveCells;
	}

	
protected:
	int noStates; /**< The number of states for the CA to consider.*/
	 
	/**
	* This method will give the new result for when setting a new state. Taking into account how many
	* bits to shift the new state.
	*@param lattice The lattice of the current CA.
	*@param newState The newState value
	*@param oldState The oldState value
	*@return The new state, bit packed containing the old state.
	*/
	__host__ __device__ int setNewState(AbstractLattice* lattice, int newState, int oldState) {
		return oldState | (newState << lattice->getNoBits());
	}

};

