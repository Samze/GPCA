#pragma once

#include "AbstractLattice.h"

class AbstractCellularAutomata
{
public:
	DLLExport AbstractCellularAutomata(void);
	DLLExport virtual ~AbstractCellularAutomata(void);
	
public:	
	//The lattice should be defined here in an abstract manner. However CUDA 1.x does not support runtime polymorphism Defines the grid and dimensions.
	//Instead we have hold the Lattice in the subclass and use a virtual method to obtain it in an OO manner.
	__host__ virtual AbstractLattice* getLattice() = 0;
	__host__ virtual void setLattice(AbstractLattice*) = 0;

	//The CA rule needs to indicate the exact size in bytes a Cell structure is.
	__host__ virtual size_t getCellSize() = 0;


};

