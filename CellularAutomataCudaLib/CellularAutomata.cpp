#include "CellularAutomata.h"

CellularAutomata::~CellularAutomata() {

	delete[] pFlatGrid;
	delete caRule;

	pFlatGrid = NULL;
	caRule = NULL;


}
CellularAutomata::CellularAutomata(DimensionType type, int dimension, int range) : dimType(type),
DIM(dimension)
{
	//initialize array based on dim with random values
	pFlatGrid = new unsigned int[DIM * DIM];

	for (int i = 0; i < DIM; ++i) {
		for (int j = 0; j < DIM; ++j) {

		////get random state value bettwen 0 & 1;
		//int random = std::rand() % range;
		////assign
		//pFlatGrid[i * DIM + j] = random == range - 1 ? 1 : 0;
		//

			
		pFlatGrid[i * DIM + j] = 1;
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

CellularAutomata::CellularAutomata(DimensionType type, unsigned int *pFlatGrid, int dimension) : dimType(type), DIM(dimension), pFlatGrid(pFlatGrid)  {

}

void CellularAutomata::generate3DGrid(int dimension, int range)
{
	dimension = DIM;
	//initialize array based on dim with random values
	pFlatGrid = new unsigned int[DIM * DIM * DIM];

	for (int i = 0; i < DIM; ++i) {
		for (int j = 0; j < DIM; ++j) {
			for (int k = 0; k < DIM; ++k) {

			//get random state value bettwen 0 & 1;
			int random = std::rand() % range;
			//assign
			pFlatGrid[(i * DIM) + j + (k * (DIM * DIM))] = random == range - 1 ? 1 : 0;
		//	pFlatGrid[(k * DIM * DIM) + (i * DIM) + j] = 0;
			}
		}
	}
}


DLLExport void CellularAutomata::setCARule(AbstractCellularAutomata* ca) {
	caRule = ca;
}