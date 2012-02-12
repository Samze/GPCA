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

#include "Abstract3DCA.h"
#include <cstdlib>


Abstract3DCA::Abstract3DCA(void)
{
}


Abstract3DCA::~Abstract3DCA(void)
{
}

DLLExport Abstract3DCA::Abstract3DCA(int dimension, int range): AbstractLattice(dimension) {
	//initialize array based on dim with random values
	pFlatGrid = new unsigned int[dimension * dimension * dimension];

	neighbourCount = new unsigned int[dimension * dimension * dimension];

	for (int i = 0; i < dimension; ++i) {
		for (int j = 0; j < dimension; ++j) {
			for (int k = 0; k < dimension; ++k) {

			//get random state value bettwen 0 & 1;
			int random = std::rand() % range;
			//assign
			pFlatGrid[(i * dimension) + j + (k * (dimension * dimension))] = random == range - 1 ? 1 : 0;
			//pFlatGrid[(k * DIM * DIM) + (i * DIM) + j] = 1;
			}
		}
	}
}

DLLExport Abstract3DCA::Abstract3DCA(unsigned int *pFlatGrid, int dimension) : AbstractLattice(dimension,pFlatGrid)  {

}