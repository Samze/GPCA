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

DLLExport Abstract2DCA::Abstract2DCA(int dimension, int range): AbstractLattice(dimension) {
	//initialize array based on dim with random values
	pFlatGrid = new unsigned int[dimension * dimension];
	
	unsigned int states = 100 * 25 * 25 * 25 * 25 * 25;

	for (int i = 0; i < dimension; ++i) {
		for (int j = 0; j < dimension; ++j) {

		////get random state value bettwen 0 & 1;
		int random = std::rand() % range;
		////assign
		//pFlatGrid[i * DIM + j] = random == range - 1 ? 1 : 0;
		//

			
		//pFlatGrid[i * dimension + j] = (50 + random) * (states/100);
		//pFlatGrid[i * dimension + j] = (50 + random) * (states/100);
	/*	if (i > DIM/2 && i < DIM/2 + 32 && j > DIM/2 && j < DIM/2 + 32) {
			pFlatGrid[i * DIM + j] = 1;
		}
		else {
			pFlatGrid[i * DIM + j] = 0;
		}*/
	
		////create cube
		//int size = 1;

		//if (i >= DIM/2 && i <= DIM/2 + size && j == DIM/2) {
		//		pFlatGrid[i * DIM + j] = 1;
		//}
		//else if (i == DIM/2 && j >= DIM/2 && j <= DIM/2 + size) {
		//		pFlatGrid[i * DIM + j] = 1;
		//}
		//else if (i == DIM/2 + size && j >= DIM/2 && j <= DIM/2 + size) {
		//		pFlatGrid[i * DIM + j] = 1;
		//}
		//else if (i >= DIM/2 && i <= DIM/2 + size && j == DIM/2 + size) {
		//		pFlatGrid[i * DIM + j] = 1;
		//}
		//
		//else {
		//	pFlatGrid[i * DIM + j] = 0;
		//}

	/*	if ((j == DIM/2 || j == DIM/2 + 1) && i == DIM/2) {
				pFlatGrid[i * DIM + j] = 1;
		}
		else {
			pFlatGrid[i * DIM + j] = 0;
		}
		*/
		}
	}
}

DLLExport Abstract2DCA::Abstract2DCA(unsigned int *pFlatGrid, int dimension) : AbstractLattice(dimension,pFlatGrid)  {

}


Abstract2DCA::~Abstract2DCA(void)
{
}
