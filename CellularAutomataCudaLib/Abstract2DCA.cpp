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

#include "Abstract2DCA.h"
#include <cstdlib>


Abstract2DCA::Abstract2DCA(void)
{
	
}

DLLExport Abstract2DCA::Abstract2DCA(int xDIM, int newYDIM, int range): AbstractLattice(xDIM) {
	//initialize array based on dim with random values
	yDIM = newYDIM;

	pFlatGrid = new unsigned int[xDIM * yDIM];

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

DLLExport Abstract2DCA::Abstract2DCA(void* pFlatGrid, int dimension) : AbstractLattice(dimension,pFlatGrid)  {

}


Abstract2DCA::~Abstract2DCA(void)
{
}
