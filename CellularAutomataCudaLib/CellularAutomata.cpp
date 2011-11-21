#include "CellularAutomataDLL.h"
#include <cstdlib>

CellularAutomata::CellularAutomata(int dimension, int range) : DIM(dimension)
{
	//initialize array based on dim with random values
	pFlatGrid = new int[DIM * DIM];

	for (int i = 0; i < DIM; ++i) {
		for (int j = 0; j < DIM; ++j) {

		////get random state value bettwen 0 & 1;
		//int random = std::rand() % range;
		////assign
		//pFlatGrid[i * DIM + j] = random == range - 1 ? 1 : 0;
		//

		//create cube
	/*	int size = 2;

		if (i >= DIM/2 && i <= DIM/2 + size && j == DIM/2) {
				pFlatGrid[i * DIM + j] = 1;
		}
		else if (i == DIM/2 && j >= DIM/2 && j <= DIM/2 + size) {
				pFlatGrid[i * DIM + j] = 1;
		}
		else if (i == DIM/2 + size && j >= DIM/2 && j <= DIM/2 + size) {
				pFlatGrid[i * DIM + j] = 1;
		}
		else if (i >= DIM/2 && i <= DIM/2 + size && j == DIM/2 + size) {
				pFlatGrid[i * DIM + j] = 0;
		}
		
		else {
			pFlatGrid[i * DIM + j] = 0;
		}*/

		if ((j == DIM/2 || j == DIM/2 + 1) && i == DIM/2) {
				pFlatGrid[i * DIM + j] = 1;
		}
		else {
			pFlatGrid[i * DIM + j] = 0;
		}
		}}
}

CellularAutomata::CellularAutomata(int *pFlatGrid, int dimension) : DIM(dimension), pFlatGrid(pFlatGrid) {}

CellularAutomata::~CellularAutomata()
{
	//clean up our allocated array
	delete [] pFlatGrid;

}