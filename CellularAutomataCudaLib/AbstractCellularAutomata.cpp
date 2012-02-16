#include "AbstractCellularAutomata.h"


AbstractCellularAutomata::AbstractCellularAutomata(void)
{
}


AbstractCellularAutomata::~AbstractCellularAutomata(void)
{
}

void AbstractCellularAutomata::setStates(unsigned int states) {

		AbstractLattice* lattice = getLattice();
		noStates = states;

		//calculate how many bits are needed to hold a states
		//we need to minus one to properly reflect the fact that 1 bit can hold 2 states
		// 3 bits can hold 8 states etc.

		states = states - 1;

		lattice->noBits = 0;
		while (states != 0) { 
			states = states >> 1; 
			++lattice->noBits;
		}

		lattice->maxBits = 1;

		for (int i = 1; i < lattice->noBits; i++) {
			lattice->maxBits = (lattice->maxBits << 1) + 1;
		}
}