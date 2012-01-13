#include "Abstract3DCA.h"


Abstract3DCA::Abstract3DCA(void)
{
}


Abstract3DCA::~Abstract3DCA(void)
{
	delete surviveNo;
	delete bornNo;
}


void Abstract3DCA::setStates(int states) {

		m_states = states;

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