#include "CellularAutomata_GPGPU.h"

CellularAutomata_GPGPU::CellularAutomata_GPGPU(int dim, int seed) : CellularAutomata(dim, seed){ }

CellularAutomata_GPGPU::~CellularAutomata_GPGPU() { }


float CellularAutomata_GPGPU::nextTimeStep(OuterTotalistic cellA) {

	return CUDATimeStep(pFlatGrid, DIM, &cellA);
}

float CellularAutomata_GPGPU::nextTimeStep(Generations cellA) {

	return CUDATimeStep(pFlatGrid, DIM, &cellA);
}