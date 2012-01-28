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
	
#include "AbstractCellularAutomata.h"

void AbstractCellularAutomata::setStates(unsigned int states) {

		noStates = states;

		//calculate how many bits are needed to hold a states
		//we need to minus one to properly reflect the fact that 1 bit can hold 2 states
		// 3 bits can hold 8 states etc.

		states = states - 1;

		noBits = 0;
		while (states != 0) { 
			states = states >> 1; 
			++noBits;
		}

		maxBits = 1;

		for (int i = 1; i < noBits; i++) {
			maxBits = (maxBits << 1) + 1;
		}
}