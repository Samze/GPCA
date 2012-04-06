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

#include "Lattice3D.h"
#include <cstdlib>


Lattice3D::Lattice3D(void)
{
}


Lattice3D::~Lattice3D(void)
{
	delete[] neighbourCount;
}

//This will create a unsigned integer array of states. 
DLLExport Lattice3D::Lattice3D(int xDIM,int newYDIM, int newZDIM,  int range): AbstractLattice(xDIM) {
	//initialize array based on dim with random values

	yDIM = newYDIM;
	zDIM = newZDIM;

	noElements = xDIM * yDIM * zDIM;

	pFlatGrid = new unsigned int[noElements];

	neighbourCount = new unsigned int[noElements];

	unsigned int* intGrid = (unsigned int*)pFlatGrid;

	for (int i = 0; i < xDIM; ++i) {
		for (int j = 0; j < yDIM; ++j) {
			for (int k = 0; k < zDIM; ++k) {

				int newVal;

				if (range == 0) {
					newVal = 0;
				} else {
					//get random state value bettwen 0 & 1;
					int random =  std::rand() % range;
					newVal = random == range - 1 ? 1 : 0;
					//assign
				}

				intGrid[(i * xDIM) + j + (k * zDIM * zDIM)] = newVal;
			//	intGrid[(k * DIM * DIM) + (i * DIM) + j] = 1;
			}
		}
	}
}

DLLExport Lattice3D::Lattice3D(void *pFlatGrid, int xSize, int ySize, int zSize) : AbstractLattice(xSize,pFlatGrid)  {

	yDIM = ySize;
	zDIM = zSize;

	noElements = xSize * ySize * zSize;

	neighbourCount = new unsigned int[noElements];
}