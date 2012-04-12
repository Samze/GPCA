#pragma once

#include "AbstractLattice.h"
#include <map>
#include <vector>

using namespace std;
/**
 * An abstract representation of a Cellular Automata rule. All rules must inherit from this class, it provides common means for
 * getting information about a CA. It is currently limited to lattice information only, but may be expanded in the future. It is limited
 * due to the nature of CUDA smx1.2 architecture limiting the use of virtual functions. Only host code make uses of the functions defined here.
 *
 */
class AbstractCellularAutomata
{
	
public:	
	
	__host__ virtual map<void**, size_t>* getDynamicArrays() = 0;

	/**
	* Virtual destructor. Forces childs destructor to be called.
	*/
	DLLExport virtual ~AbstractCellularAutomata(void);
	
	/**
	* Gets the lattice for the CA rule. This method is implemented by a subclass.
	*@return Returns the currnetly assigned lattice.
	*/
	__host__ virtual AbstractLattice* getLattice() = 0;
	
	/**
	* Sets the lattice for this CA rule. This method is implemented by a subclass. Note any implementation of this
	* class should delete the existing lattice.
	*@param newLattice The new lattice to be set.
	*/
	__host__ virtual void setLattice(AbstractLattice* newLattice) = 0;

private:
	//The lattice should be defined here in an abstract manner. However CUDA 1.x does not support runtime polymorphism Defines the grid and dimensions.
	//Instead we have hold the Lattice in the subclass and use a virtual method to obtain it in an OO manner.
	//AbstractLattice* lattice;
};

