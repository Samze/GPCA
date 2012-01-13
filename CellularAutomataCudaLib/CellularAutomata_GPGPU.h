#ifndef CELLULARAUTOMATA_GPGPU_DLL_H
#define CELLULARAUTOMATA_GPGPU_DLL_H

#include "CellularAutomata.h"
#include "CellularAutomata_launcher.cu"

#include "OuterTotalistic.h"
#include "Generations.h"
//#include "Generations3D.h"



#define DLLExport __declspec(dllexport)

//forward declaration.
//template<typename CAFunction> extern float CUDATimeStep(unsigned int* pFlatGrid, int DIM, CAFunction *func);

class CellularAutomata_GPGPU : public CellularAutomata
{
public:
	DLLExport CellularAutomata_GPGPU(int, int);
	DLLExport CellularAutomata_GPGPU(unsigned int*, int);
	DLLExport ~CellularAutomata_GPGPU();
	
	float nextTimeStep();

};

#endif

