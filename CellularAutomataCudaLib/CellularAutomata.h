#ifndef CELLULARAUTOMATA_DLL_H
#define CELLULARAUTOMATA_DLL_H

#include <cstdlib>

#include "OuterTotalistic.h"
#include "Generations.h"
#include "Generations3D.h"

#include <cuda_runtime.h>
#include <cutil_inline.h>
#include <cutil_gl_inline.h>
#include <cuda_gl_interop.h>

#define DLLExport __declspec(dllexport)

class CellularAutomata
{

public:
	
	enum DimensionType
	{
		TWO_D,
		THREE_D
	};

	DLLExport CellularAutomata(DimensionType,int,int); //random data
	DLLExport CellularAutomata(DimensionType,unsigned int*, int);
	DLLExport ~CellularAutomata();
	
	DLLExport virtual float nextTimeStep() = 0;
	
	DLLExport unsigned int getDIM() { return DIM;};
	DLLExport unsigned int* getGrid() { return pFlatGrid;};

	DLLExport AbstractCellularAutomata* getCARule() { return caRule; };
	DLLExport void setCARule(AbstractCellularAutomata* ca);
	DLLExport void generate3DGrid(int,int);
	
	DimensionType dimType;

	
	DLLExport virtual unsigned int initCudaForGL() = 0;
	DLLExport virtual void cudaBindPDO(GLuint* pbo) = 0;
	DLLExport virtual void cudaUnBindPDO(GLuint* pbo) = 0;
	DLLExport virtual void runCuda(GLuint* pbo) = 0;
	DLLExport virtual void launch_kernalPDO2(uchar4* pos,unsigned int w,unsigned int h) = 0;

protected :
	const unsigned DIM; //Should make const?
	unsigned int *pFlatGrid;
	AbstractCellularAutomata* caRule;
};

#endif // CELLULARAUTOMATA_DLL_H
