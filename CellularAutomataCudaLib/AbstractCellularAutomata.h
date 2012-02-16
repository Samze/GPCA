#pragma once

#include "AbstractLattice.h"

class AbstractCellularAutomata
{
public:
	DLLExport AbstractCellularAutomata(void);
	DLLExport virtual ~AbstractCellularAutomata(void);
	
	DLLExport __host__ virtual void setStates(unsigned int states);
	DLLExport __host__ int getNoStates() { return noStates;}

	__host__ __device__ int setNewState(AbstractLattice* lattice, int newState, int oldState) {

		return oldState | (newState << lattice->noBits);
	
	}

public:	
	//The lattice should be defined here in an abstract manner. However CUDA 1.x does not support runtime polymorphism Defines the grid and dimensions.
	//Instead we have hold the Lattice in the subclass and use a virtual method to obtain it in an OO manner.
	__host__ virtual AbstractLattice* getLattice() = 0;

protected:
	int noStates;

};

