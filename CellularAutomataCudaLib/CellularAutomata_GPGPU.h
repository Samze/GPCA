#ifndef CELLULARAUTOMATA_GPGPU_DLL_H
#define CELLULARAUTOMATA_GPGPU_DLL_H

#include "CellularAutomataDLL.h"
#include "CellularAutomata_launcher.cu"

#include "OuterTotalistic.h"
#include "Generations.h"


#define DLLExport __declspec(dllexport)

//forward declaration.
template<typename CAFunction> extern float CUDATimeStep(int* pFlatGrid, int DIM, CAFunction *func);

class CellularAutomata_GPGPU : public CellularAutomata
{
public:
	DLLExport CellularAutomata_GPGPU(int, int);
	DLLExport ~CellularAutomata_GPGPU();

	float nextTimeStep(OuterTotalistic);
	float nextTimeStep(Generations);
};

#endif

