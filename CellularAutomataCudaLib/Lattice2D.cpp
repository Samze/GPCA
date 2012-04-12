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

#include "Lattice2D.h"
#include <cstdlib>


Lattice2D::Lattice2D(void)
{
	
}

Lattice2D::Lattice2D(void* grid, int sizeX, int sizeY) : AbstractLattice(sizeX,pFlatGrid)  {

	yDIM = sizeY;

	noElements = xDIM * yDIM;
	pFlatGrid = grid;
}

DLLExport Lattice2D::Lattice2D(int xDIM, int newYDIM, int range): AbstractLattice(xDIM) {
	//initialize array based on dim with random values
	yDIM = newYDIM;

	noElements = xDIM * yDIM;

	pFlatGrid = new unsigned int[noElements];

	unsigned int* intGrid = (unsigned int*)pFlatGrid;

	for (int i = 0; i < xDIM; ++i) {
		for (int j = 0; j < yDIM; ++j) {

			int newVal;

			if(range == 0) {
				newVal = 0;
			}
			else {
				//get random state value bettwen 0 & 1;
				int random = std::rand() % range;
				newVal = random == range - 1 ? 1 : 0;
			}
		////assign
		intGrid[i * yDIM + j] = newVal;
		//
		}
	}
}
//
//DLLExport Lattice2D::Lattice2D(void* pFlatGrid, int dimension) : AbstractLattice(dimension,pFlatGrid)  {
//
//}


Lattice2D::~Lattice2D(void)
{
}
