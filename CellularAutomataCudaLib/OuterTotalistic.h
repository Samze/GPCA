/*	GPCA - A Cellular Automata library powered by CUDA. 
    Copyright (C) 2011  Sam Gunaratne University of Plymouth

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.*/
#define OUTER_TOTALISTIC_H_

#include "device_launch_parameters.h"
#include "Abstract2DCA.h"
#include "Totalistic.h"

class OuterTotalistic : public Abstract2DCA {

public :
	DLLExport __device__ __host__ OuterTotalistic() {}
	DLLExport __device__ __host__ ~OuterTotalistic() {}
	

	__device__ __host__ int applyFunction(unsigned int* g_data, int x, int y, int xDIM) { 
		
		int state = g_data[x * xDIM + y];
		
		int neighbourhoodStates[8];
	
		//set as -1 by default.
		for(int i = 0; i < 8; i++) {
			neighbourhoodStates[i] = -1; 
		}

		getNeighbourhood(neighbourhoodStates,g_data,x * xDIM,y,xDIM);

		int liveCells = Totalistic::getLiveCellCount(neighbourhoodStates,maxBits,neighbourhoodType);

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
