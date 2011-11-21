#ifndef OUTER_TOTALISTIC_H_
#define OUTER_TOTALISTIC_H_

#include "device_launch_parameters.h"
#include "Abstract2DCA.h"

class OuterTotalistic : public Abstract2DCA {

public :
	DLLExport __device__ __host__ OuterTotalistic() {}
	DLLExport __device__ __host__ ~OuterTotalistic() {}

	__device__ __host__ int applyFunction(int* g_data, int x, int y, int xDIM) { 
		
		int state = g_data[x * xDIM + y];
		
		int liveCells =  getNeighbourhood(g_data, x * xDIM, y, xDIM, neighbourhoodType);

		for (int i = 0; i < surviveSize; i++) {
			if (state && liveCells == surviveNo[i]) return state | (1 << noBits);
		}
		
		for (int i = 0; i < bornSize; i++) {		
			if (!state && liveCells == bornNo[i]) return state |  (1 << noBits);
		}

		return state; 
	}
};


#endif
