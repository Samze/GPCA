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

#ifndef CELLULARAUTOMATA_GPGPU_DLL_H
#define CELLULARAUTOMATA_GPGPU_DLL_H

#include "CellularAutomata.h"
#include "CellularAutomata_launcher.cu"

//#include "Generations3D.h"

#define DLLExport __declspec(dllexport)

//forward declaration.
//extern "C" __global__ void kernalBufferObjectTest(GLfloat* pos,unsigned int w,unsigned int h);

/**
* Represents a GPU implemented of running a new Cellular Automata. This will use a parallel implementation applying the transition function
* toall cells in a simulatanious manner.
* @see CellularAutomata_GPGPU
*/
class CellularAutomata_GPGPU : public CellularAutomata
{
public:
	/**
	* Default constructor.
	*/
	DLLExport CellularAutomata_GPGPU();

	/**
	* Destructor.
	*/
	DLLExport ~CellularAutomata_GPGPU();
	
	/**
	* GPU version of the timestep.
	* @returns The time taken by the GPU to process the transition function to all cells in the lattice.
	*/ 
	float nextTimeStep();
};
#endif