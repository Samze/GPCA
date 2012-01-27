#ifndef CELLULARAUTOMATA_GPGPU_DLL_H
#define CELLULARAUTOMATA_GPGPU_DLL_H

#include "CellularAutomata.h"
#include "CellularAutomata_launcher.cu"

#include "OuterTotalistic.h"
#include "Generations.h"
//#include "Generations3D.h"

#define DLLExport __declspec(dllexport)

//forward declaration.
extern "C" __global__ void kernalBufferObjectTest(GLfloat* pos,unsigned int w,unsigned int h);
//template<typename CAFunction> extern float CUDATimeStep(unsigned int* pFlatGrid, int DIM, CAFunction *func);
//template<typename CAFunction> extern float CUDATimeStep3D(unsigned int* pFlatGrid, int DIM, CAFunction *func);

class CellularAutomata_GPGPU : public CellularAutomata
{
public:
	DLLExport CellularAutomata_GPGPU(DimensionType type, int, int);
	DLLExport CellularAutomata_GPGPU(DimensionType type, unsigned int*, int);
	DLLExport ~CellularAutomata_GPGPU();
	
	float nextTimeStep();

	GLuint pbo;
	GLuint textureID;

	
	DLLExport unsigned int initCudaForGL();
	DLLExport void cudaBindPDO(GLuint pbo);
	DLLExport void cudaUnBindPDO(GLuint pbo);
	DLLExport void runCuda(GLuint pbo);
	void launch_kernalPDO2(GLfloat* pos,unsigned int w,unsigned int h);
	
	const char* errorCheck();
	struct cudaGraphicsResource* positionsVBO_CUDA;
};
#endif