#ifndef CELLULARAUTOMATA_DLL_H
#define CELLULARAUTOMATA_DLL_H

#include <cstdlib>

#include "OuterTotalistic.h"
#include "Generations.h"

#define DLLExport __declspec(dllexport)

class CellularAutomata
{

public:
	DLLExport CellularAutomata(int,int); //random data
	DLLExport CellularAutomata(int*, int);
	DLLExport ~CellularAutomata();
	//DLLExport virtual float nextTimeStep() = 0;
	
	DLLExport virtual float nextTimeStep(OuterTotalistic) = 0;
	DLLExport virtual float nextTimeStep(Generations) = 0;
	
	DLLExport unsigned int getDIM() { return DIM;};
	DLLExport int* getGrid() { return pFlatGrid;};

protected :
	const unsigned DIM; //Should make const?
	int *pFlatGrid;

};

#endif // CELLULARAUTOMATA_DLL_H
