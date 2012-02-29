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
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include "device_launch_parameters.h"

class Totalistic {

public:
	__device__  __host__ Totalistic(void) { }
	__device__  __host__ ~Totalistic(void) { }

	__device__ __host__  static unsigned int getLiveCellCount(int* neighbourhoodStates, int maxBits, int neighbourType) {

		unsigned int numLiveCells =0;

		for(int i = 0; i < neighbourType; ++i) {
			if(neighbourhoodStates[i] != -1) 
				if((neighbourhoodStates[i] & maxBits) == 1) //This cell's state is alive.
					++numLiveCells;
		}

		return numLiveCells;
	}
};

