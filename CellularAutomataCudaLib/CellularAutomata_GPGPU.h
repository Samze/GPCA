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

#include "OuterTotalistic.h"
#include "Generations.h"
#include "SCIARA.h"
//#include "Generations3D.h"

#define DLLExport __declspec(dllexport)

//forward declaration.
extern "C" __global__ void kernalBufferObjectTest(GLfloat* pos,unsigned int w,unsigned int h);
//template<typename CAFunction> extern float CUDATimeStep(unsigned int* pFlatGrid, int DIM, CAFunction *func);
//template<typename CAFunction> extern float CUDATimeStep3D(unsigned int* pFlatGrid, int DIM, CAFunction *func);

class CellularAutomata_GPGPU : public CellularAutomata
{
public:
	DLLExport CellularAutomata_GPGPU();
	DLLExport ~CellularAutomata_GPGPU();
	
	float nextTimeStep();
};
#endif