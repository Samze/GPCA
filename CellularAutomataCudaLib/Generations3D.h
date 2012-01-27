#ifndef GENERATIONS3D_H
#define GENERATIONS3D_H

#include "device_launch_parameters.h"
#include "Abstract3DCA.h"
#include "cuda.h"

class Generations3D : public Abstract3DCA {

public :
	DLLExport __device__ __host__ Generations3D() {}
	DLLExport __device__ __host__ ~Generations3D() {}

	__device__  int applyFunction(unsigned int* g_data, int x, int y, int z, int xDIM) { 
		
		int xAltered = x * xDIM;
		int zAltered = z * xDIM * xDIM;

		int state = g_data[zAltered + xAltered + y];
		int temp = 0;

		//We want know about neighbours even if we're not using them to set the next state, this is 
		//so they can not be rendered by the viewer. To speed up the processing, move this to the else

		int neighbourhoodStates[26];
	
		//set as -1 by default.
		for(int i = 0; i < 26; i++) {
			neighbourhoodStates[i] = -1; 
		}

		//Populate neighbours states.
		getNeighbourhood(neighbourhoodStates,g_data, xAltered, y, zAltered, xDIM);

		//we only care about neighbours when we know we're in a ready state
		//int liveCells =  getNeighbourhood(g_data, x * xDIM, y, xDIM, neighbourhoodType);
		int liveCells = Totalistic::getLiveCellCount(neighbourhoodStates,maxBits,neighbourhoodType);

		//int liveCells = getNeighbourhood(g_data, xAltered, y, zAltered, xDIM);

		//This is for culling of cubes surrounded on all sides.
		neighbourCount[xAltered + y + zAltered] = liveCells;

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

			for (int i = 0; i < surviveSize; i++) {
				if (state == 1 && liveCells == surviveNo[i]) return state | (1 << noBits);
			}
			
			for (int i = 0; i < bornSize; ++i) {		
				if (state == 0 && liveCells == bornNo[i]) return state | (1 << noBits);
			}
			
			if (state == 1) {
				if (state < noStates - 1) { //This guards against 2 state generations
					return state | (2 << noBits);
				}
			}

		}
		
		return state;
	}
};


#endif
