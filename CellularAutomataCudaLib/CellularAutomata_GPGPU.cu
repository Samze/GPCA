#include "CellularAutomata_GPGPU.h"

CellularAutomata_GPGPU::CellularAutomata_GPGPU(int dim, int seed) : CellularAutomata(dim, seed){ }


CellularAutomata_GPGPU::CellularAutomata_GPGPU(unsigned int* pFlatGrid, int seed) : CellularAutomata(pFlatGrid, seed){ }

CellularAutomata_GPGPU::~CellularAutomata_GPGPU() { 


}

float CellularAutomata_GPGPU::nextTimeStep() {
	
	Generations* v = dynamic_cast<Generations*>(caRule);
	OuterTotalistic* v2 = dynamic_cast<OuterTotalistic*>(caRule);

	if(v != 0) {
		return CUDATimeStep(pFlatGrid, DIM, v);
	}
	else if (v2 != 0) {
		return CUDATimeStep(pFlatGrid, DIM, v2);
	}

	return -1;
}

float CellularAutomata_GPGPU::nextTimeStep(OuterTotalistic cellA) {

	return CUDATimeStep(pFlatGrid, DIM, &cellA);
}

float CellularAutomata_GPGPU::nextTimeStep(Generations cellA) {

	return CUDATimeStep(pFlatGrid, DIM, &cellA);
}