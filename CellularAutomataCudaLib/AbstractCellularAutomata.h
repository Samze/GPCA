#ifndef ABSTRACT_CELLULAR_AUTOMATA_H
#define ABSTRACT_CELLULAR_AUTOMATA_H

#include "device_launch_parameters.h"

#define DLLExport __declspec(dllexport)

class AbstractCellularAutomata
{

public:
	DLLExport AbstractCellularAutomata(void) {}
	DLLExport ~AbstractCellularAutomata(void) {}
	
	//This next line should be here to provide 'proper' virtual inheritence, sadly it is only supported on CUDA sm_2x architecture.
	//__device__ __host__ int applyFunction(int*,int,int,int) {
	//	return 3;
	//}

	__device__ int getX() {
		return testX;
	}

private:
	int testX;
};


#endif