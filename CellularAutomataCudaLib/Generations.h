#ifndef GENERATIONS_H
#define GENERATIONS_H

#include "device_launch_parameters.h"
#include "Abstract2DCA.h"
#include "cuda.h"

class Generations : public Abstract2DCA {

public :
	DLLExport __device__ __host__ Generations() {}
	DLLExport __device__ __host__ ~Generations() {}

	__device__  int applyFunction(int* g_data, int x, int y, int xDIM) { 
		
		int state = g_data[x * xDIM + y];
		int temp = 0;
		//generations specialism
		if (state >= 1) {
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
			int liveCells =  getNeighbourhood(g_data, x * xDIM, y, xDIM, neighbourhoodType);
		
			//for (int i = 0; i < surviveSize; i++) {
			//	if (state && liveCells == surviveNo[i]) return state | (1 << noBits);;
			//}
			//

			for (int i = 0; i < bornSize; ++i) {		
				if (state == 0 && liveCells == bornNo[i]) return state | (1 << noBits);;
			}
		}
		return state;
	}
};


#endif
