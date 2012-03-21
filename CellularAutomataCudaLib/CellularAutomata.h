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
#include "Generations.h"
#include "Generations3D.h"
#include "SCIARA.h"
#include "SCIARA2.h"

#include <cuda_runtime.h>
#include <cutil_inline.h>
#include <cutil_gl_inline.h>
#include <cuda_gl_interop.h>
#include <stdlib.h>

#define DLLExport __declspec(dllexport)

class CellularAutomata
{

public:

	DLLExport CellularAutomata();
	DLLExport ~CellularAutomata();
	
	DLLExport virtual float nextTimeStep() = 0;
	
//	DLLExport unsigned int* getGrid() { return pFlatGrid;};

	DLLExport AbstractCellularAutomata* getCARule() { return caRule; };
	DLLExport void setCARule(AbstractCellularAutomata* ca);
	DLLExport void generate3DGrid(int,int);

	DLLExport std::string getRuleName(){return ruleName;}

protected :
	//unsigned int *pFlatGrid;
	AbstractCellularAutomata* caRule;
	std::string ruleName; //Meta data about what class we're launching
};

#endif // CELLULARAUTOMATA_DLL_H
