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

	cudaGLSetGLDevice(cutGetMaxGflopsDeviceId());
	errorCheck();
	return 1;
}

void CellularAutomata_GPGPU::cudaBindPDO(GLuint pbo) {

	//cudaGraphicsGLRegisterBuffer(ap
	//cudaGLRegisterBufferObject(pbo);
	cudaGraphicsGLRegisterBuffer(&positionsVBO_CUDA,pbo,cudaGraphicsMapFlagsWriteDiscard);
	errorCheck();
}

void CellularAutomata_GPGPU::cudaUnBindPDO(GLuint pbo) {
	//cudaGLUnregisterBufferObject(pbo);

	cudaGraphicsUnregisterResource(positionsVBO_CUDA);
	
	errorCheck();
}

void CellularAutomata_GPGPU::runCuda(GLuint pbo) {

	GLfloat *dev_ptr = NULL; 
	size_t numBytes;

	cudaGraphicsMapResources(1,&positionsVBO_CUDA,0);
	errorCheck();

	//cudaGLMapBufferObject((void**)&dev_ptr,pbo);
	cudaGraphicsResourceGetMappedPointer((void**)&dev_ptr,&numBytes,positionsVBO_CUDA);
	errorCheck();

	launch_kernalPDO2(dev_ptr,100,100);
	errorCheck();

	//cudaGLUnmapBufferObject(pbo);
	cudaGraphicsUnmapResources(1,&positionsVBO_CUDA,0);
	errorCheck();
}


//Todo, move this somewhere useful
const char* CellularAutomata_GPGPU::errorCheck() {
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		fprintf(stderr, "C error : %s",err);
		return cudaGetErrorString(err);
	}
	return  NULL;
}

void CellularAutomata_GPGPU::launch_kernalPDO2(GLfloat* pos,unsigned int w,unsigned int h) {

	dim3 block(6,4,1);
	dim3 grid(1,1,1);

	size_t size = sizeof(GLfloat) * 24;

	kernalBufferObjectTest<<<grid,block>>>(pos,w,h);

	cudaThreadSynchronize();

	errorCheck();
}


extern "C" __global__ void kernalBufferObjectTest(GLfloat* pos,unsigned int w,unsigned int h) {

	int index = threadIdx.y * blockDim.x + threadIdx.x;

	if(index < 24) {
		pos[index] = pos[index] + 0.1;
	}
}