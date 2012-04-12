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

#ifndef CELLULARAUTOMATA_DLL_H
#define CELLULARAUTOMATA_DLL_H

#include "OuterTotalistic.h"
#include "OuterTotalistic3D.h"
#include "Generations.h"
#include "Generations3D.h"
#include "SCIARA.h"
#include "SCIARAThickness.h"

#include <cuda_runtime.h>
#include <cutil_inline.h>
#include <cutil_gl_inline.h>
#include <cuda_gl_interop.h>
#include <stdlib.h>

#define DLLExport __declspec(dllexport)

/**
* Abstract class for launching a Cellular Automata. This class is the container and launcher for a CA. It contains the rule and it's subclasses
* provide ways of running the nextTimeStep function.
*
*/
class CellularAutomata
{

public:

	/**
	* Default constructor.
	*/
	DLLExport CellularAutomata();
	
	/**
	* Virtual destructor. This class is meant to be subclassed.
	*/
	DLLExport virtual ~CellularAutomata();
	
	/**
	* This applies the set rule to the lattice value for one time step. This can be seen as CA = func(CA,t+1). This is the method used to 
	* activate the running of a CA.
	*/
	DLLExport virtual float nextTimeStep() = 0;

	/**
	* Gets the currently assigned CA rule.
	*@return Returns the currently set CA rule.
	*/
	DLLExport AbstractCellularAutomata* getCARule() { return caRule; };
	
	/**
	* Sets the currently assigned CA rule.
	*@param ca The new CA rule to be assigned.
	*/
	DLLExport void setCARule(AbstractCellularAutomata* ca);

	/**
	* Provides utilities to know which CA rule is currently bound.
	*@return The Cellular automata rule name.
	*/
	DLLExport std::string getRuleName(){return ruleName;}

	/**
	* Gets the number of times the function has been applied to the lattice. The value of T.
	*@return The number of iterations run thus far.
	*/
	DLLExport int getStepNumber();

	/**
	* Sets the number of times the function has been applied to the lattice. The value of T. This can be used if a new (previous)
	* lattice has been applied.
	*@param num The number of iterations run thus far.
	*/
	DLLExport void setStepNumber(int num);

protected :
	AbstractCellularAutomata* caRule;  /**< The CA defined.*/
	std::string ruleName; /**< Meta data about what class we're launching, this should probably be encapsulated in the AbstractCA class.*/
	int stepNumber;  /**< The value of t. The number of steps that have been run. */
};

#endif // CELLULARAUTOMATA_DLL_H
