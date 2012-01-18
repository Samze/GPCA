#ifndef GENERATIONS3D_H
#define GENERATIONS3D_H

#include "device_launch_parameters.h"
#include "Abstract3DCA.h"
#include "cuda.h"

class Generations3D : public Abstract3DCA {

public :
	DLLExport __device__ __host__ Generations3D() {}
	DLLExport __device__ __host__ ~Generations3D() {}
	virtual void test() {}

	__device__  int applyFunction(unsigned int* g_data, int x, int y, int z, int xDIM) { 
		
		int xAltered = x * xDIM;
		int zAltered = z * xDIM * xDIM;

		int state = g_data[zAltered + xAltered + y];
		int temp = 0;
		
			//we only care about neighbours when we know we're in a ready state
			int liveCells = getNeighbourhood(g_data, xAltered, y, zAltered, xDIM, neighbourhoodType);
	
			neighbourCount[xAltered + y + zAltered] = liveCells;

			for (int i = 0; i < surviveSize; i++) {
				if (state == 1 && liveCells == surviveNo[i]) return state | (1 << noBits);
			}
			
			for (int i = 0; i < bornSize; ++i) {		
				if (state == 0 && liveCells == bornNo[i]) return state | (1 << noBits);
			}

		return state;

	}
};


#endif
