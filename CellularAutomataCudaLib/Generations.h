#ifndef GENERATIONS_H
#define GENERATIONS_H

#include "device_launch_parameters.h"
#include "Abstract2DCA.h"
#include "cuda.h"
#include "Totalistic.h"

class Generations : public Abstract2DCA{

public :
	DLLExport __device__ __host__ Generations() {}
	DLLExport __device__ __host__ ~Generations() {}

	__device__  int applyFunction(unsigned int* g_data, int x, int y, int xDIM) { 
		
		int state = g_data[x * xDIM + y];
		int temp = 0;

		//generations specialism
		if (state > 1) {
			if(state >= noStates - 1) {
				//reset this state next go
				return state;
			}
			else {
				temp = state + 1;
				return state | ((temp) << noBits);
			}
		}
		else {
			//This is signed so we can default to -1, this shows NO neighbour, a result of 0 means a neighbour who's state is zero
			//This may cause complications as states are stored in unsigned ints...
			int neighbourhoodStates[8];
	
			//set as -1 by default.
			for(int i = 0; i < 8; i++) {
				neighbourhoodStates[i] = -1; 
			}

			//Populate neighbours states.
			getNeighbourhood(neighbourhoodStates,g_data,x * xDIM,y,xDIM);

			//we only care about neighbours when we know we're in a ready state
			//int liveCells =  getNeighbourhood(g_data, x * xDIM, y, xDIM, neighbourhoodType);
			int liveCells = Totalistic::getLiveCellCount(neighbourhoodStates,maxBits,neighbourhoodType);
	
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
