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
	DLLExport CellularAutomata(unsigned int*, int);
	DLLExport ~CellularAutomata();
	//DLLExport virtual float nextTimeStep() = 0;
	
	DLLExport virtual float nextTimeStep() = 0;
	DLLExport virtual float nextTimeStep(OuterTotalistic) = 0;
	DLLExport virtual float nextTimeStep(Generations) = 0;
	
	DLLExport unsigned int getDIM() { return DIM;};
	DLLExport unsigned int* getGrid() { return pFlatGrid;};

	DLLExport Abstract2DCA* getCARule() { return caRule; };
	DLLExport void setCARule(Abstract2DCA* ca) { caRule = ca;};

protected :
	const unsigned DIM; //Should make const?
	unsigned int *pFlatGrid;
	Abstract2DCA* caRule;

};

#endif // CELLULARAUTOMATA_DLL_H
