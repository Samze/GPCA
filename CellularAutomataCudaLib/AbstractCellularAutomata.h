#ifndef ABSTRACT_CELLULAR_AUTOMATA_H
#define ABSTRACT_CELLULAR_AUTOMATA_H

#include "device_launch_parameters.h"

#define DLLExport __declspec(dllexport)

class AbstractCellularAutomata
{

public:
	DLLExport AbstractCellularAutomata(void) {}
	DLLExport virtual ~AbstractCellularAutomata(void) {}
	
	DLLExport __host__ void setStates(unsigned int states);
	DLLExport __host__ int getNoStates() { return noStates;}
	__host__ int getNoBits() { return noBits;}

protected:
	int noStates;
	int noBits;
	int maxBits;

	//This next line should be here to provide 'proper' virtual inheritence, sadly it is only supported on CUDA sm_2x architecture.
	//__device__ __host__ int applyFunction(int*,int,int,int) {
	//	return 3;
	//}

};


#endif