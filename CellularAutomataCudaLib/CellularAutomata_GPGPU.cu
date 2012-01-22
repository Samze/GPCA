#include "CellularAutomata_GPGPU.h"


CellularAutomata_GPGPU::CellularAutomata_GPGPU(DimensionType type, int dim, int seed) : CellularAutomata(type, dim, seed){ }


CellularAutomata_GPGPU::CellularAutomata_GPGPU(DimensionType type, unsigned int* pFlatGrid, int seed) : CellularAutomata(type, pFlatGrid, seed){ }

CellularAutomata_GPGPU::~CellularAutomata_GPGPU() { 


}

float CellularAutomata_GPGPU::nextTimeStep() {
	
	Generations* v = dynamic_cast<Generations*>(caRule);
	OuterTotalistic* v2 = dynamic_cast<OuterTotalistic*>(caRule);
	Generations3D* v3 = dynamic_cast<Generations3D*>(caRule);

	if(v != 0) {
		return CUDATimeStep(pFlatGrid, DIM, v);
	}
	else if (v2 != 0) {
		return CUDATimeStep(pFlatGrid, DIM, v2);
	}
	else if(v3 != 0) {
		return CUDATimeStep3D(pFlatGrid,DIM,v3);
	}
	return -1;
}



unsigned int  CellularAutomata_GPGPU::initCudaForGL() {

	
	//cudaGLSetGLDevice(cutGetMaxGflopsDeviceId());

	return 1;
}

void CellularAutomata_GPGPU::cudaBindPDO(GLuint* pbo) {

	//cudaGLRegisterBufferObject(*pbo);

}

void CellularAutomata_GPGPU::cudaUnBindPDO(GLuint* pbo) {
	//cudaGLUnregisterBufferObject(*pbo);

}

void CellularAutomata_GPGPU::runCuda(GLuint* pbo) {

	uchar4 *dev_ptr = NULL;

	//cudaGLMapBufferObject((void**)&dev_ptr,*pbo);

	//launch_kernalPDO2(dev_ptr,100,100);

//	cudaGLUnmapBufferObject(*pbo);

}

extern "C" void CellularAutomata_GPGPU::launch_kernalPDO2(uchar4* pos,unsigned int w,unsigned int h) {


}