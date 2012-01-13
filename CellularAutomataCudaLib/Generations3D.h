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

	__device__  int applyFunction(unsigned int* g_data, int x, int y, int z,int xDIM) { 
		
		int state = g_data[(z * xDIM * xDIM) + (x * xDIM) + y];
		int temp = 0;

		//generations specialism
		if (state > 1) {
			if(state >= m_states - 1) {
				//reset this state next go
				return state;
			}
			else {
				temp = state + 1;
				return state | ((temp) << noBits);
			}
		}
		else {
		
			//we only care about neighbours when we know we're in a ready state
			int liveCells = getNeighbourhood(g_data, x * xDIM, y, z * xDIM *xDIM, xDIM, neighbourhoodType);
	
			for (int i = 0; i < surviveSize; i++) {
				if (state == 1 && liveCells == surviveNo[i]) return state | (1 << noBits);
			}
			
			for (int i = 0; i < bornSize; ++i) {		
				if (state == 0 && liveCells == bornNo[i]) return state | (1 << noBits);
			}
			
			if (state == 1) return state | (2 << noBits);
		}

		return state;

	}
};


#endif
