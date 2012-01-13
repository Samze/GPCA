#pragma once

#include "device_launch_parameters.h"
#include "AbstractCellularAutomata.h"

class Abstract3DCA : public AbstractCellularAutomata
{
public:
	DLLExport Abstract3DCA(void);
	DLLExport ~Abstract3DCA(void);
	virtual void test() = 0;

	int m_states;

	int noBits;
	int maxBits;

    int* surviveNo;
    int  surviveSize;

	int* bornNo;
    int bornSize;

	int neighbourhoodType;
	//Always better to use constants than defines (effective C++)
	static const int THREEDMOORE = 0;
	
	DLLExport void setStates(int);
	DLLExport int getNoStates() { return m_states;};

	__device__ __host__ void setSurviveNo(int* list, int size) {
		surviveNo = list;
		surviveSize = size;
	}

	__device__ __host__ void setBornNo(int* list, int size) {
		bornNo = list;
		bornSize = size;
	}
	
	__device__ __host__ int getNeighbourhood(unsigned int* g_data, int x, int y, int z, int xDIM, int neighbourhood) {

		switch(neighbourhood) {
		case THREEDMOORE:
			return get3dMooresNeighbourhood(g_data,x,y,z,xDIM);
		default:
			return 0;
		}
	}

	//probably a much better way to figure out the moores neighbourhood
	__device__ __host__ int get3dMooresNeighbourhood(unsigned int* g_data, int x, int y, int z, int xDIM) {

		//get neighbours for cell x,y
		int numlivecells = 0;

		int zDIM = xDIM * xDIM;

		//z = -1

		// [-1,-1]
		if (x != 0 && y != 0 && z != 0)
			if ((g_data[x - xDIM + y - 1 + z - zDIM] & maxBits) == 1)
				++numlivecells;

		// [0,-1]
		if ( y != 0 && z != 0)
			if ((g_data[x + y - 1 + z - zDIM] & maxBits) == 1)
				++numlivecells;

		// [1,-1]
		if (x != xDIM - 1 && y != 0 && z != 0)
			if ((g_data[x + xDIM + y - 1 + z - zDIM] & maxBits) == 1)
				++numlivecells;

		// [-1,0]
		if (x != 0 && z != 0)
			if ((g_data[x - xDIM + y + z - zDIM] & maxBits)  == 1)
				++numlivecells;	

		// [0,0]
		if (z != 0)
			if ((g_data[x + y + z - zDIM] & maxBits) == 1)
				++numlivecells;

		// [1,0]
		if (x != xDIM - 1 && z != 0)
			if ((g_data[x + xDIM + y + z - zDIM] & maxBits) == 1)
				++numlivecells;

		// [-1,1]
		if (x != 0 && y != xDIM -1 && z != 0 )
			if ((g_data[x - xDIM + y + 1 + z - zDIM] & maxBits) == 1)
				++numlivecells;

		// [0,1]
		if (y != xDIM -1 && z != 0 )
			if ((g_data[x + y + 1 + z - zDIM] & maxBits) == 1)
				++numlivecells;

		// [1,1]
		if (x != xDIM -1 && y != xDIM - 1 && z != 0 )
			if ((g_data[x + xDIM + y + 1 + z - zDIM] & maxBits) == 1)
				++numlivecells;

		//z = 0

		// [-1,-1]
		if (x != 0 && y != 0)
			if ((g_data[x - xDIM + y - 1 + z] & maxBits) == 1)
				++numlivecells;

		// [0,-1]
		if ( y != 0)
			if ((g_data[x + y - 1  + z] & maxBits) == 1)
				++numlivecells;

		// [1,-1]
		if (x != xDIM - 1 && y != 0 )
			if ((g_data[x + xDIM + y - 1  + z] & maxBits) == 1)
				++numlivecells;

		// [-1,0]
		if (x != 0)
			if ((g_data[x - xDIM + y  + z] & maxBits)  == 1)
				++numlivecells;	

		// [1,0]
		if (x != xDIM - 1)
			if ((g_data[x + xDIM + y  + z] & maxBits) == 1)
				++numlivecells;

		// [-1,1]
		if (x != 0 && y != xDIM -1 )
			if ((g_data[x - xDIM + y + 1  + z] & maxBits) == 1)
				++numlivecells;

		// [0,1]
		if (y != xDIM -1 )
			if ((g_data[x + y + 1  + z] & maxBits) == 1)
				++numlivecells;

		// [1,1]
		if (x != xDIM -1 && y != xDIM - 1 )
			if ((g_data[x + xDIM + y + 1  + z] & maxBits) == 1)
				++numlivecells;

		//z = 1
		
		// [-1,-1]
		if (x != 0 && y != 0 && z != zDIM - 1)
			if ((g_data[x - xDIM + y - 1 + z + zDIM ] & maxBits) == 1)
				++numlivecells;

		// [0,-1]
		if ( y != 0  && z != zDIM - 1)
			if ((g_data[x + y - 1 + z + zDIM ] & maxBits) == 1)
				++numlivecells;

		// [1,-1]
		if (x != xDIM - 1 && y != 0  && z != zDIM - 1)
			if ((g_data[x + xDIM + y - 1 + z + zDIM ] & maxBits) == 1)
				++numlivecells;

		// [-1,0]
		if (x != 0  && z != zDIM - 1)
			if ((g_data[x - xDIM + y + z + zDIM ] & maxBits)  == 1)
				++numlivecells;	
		
		// [0,0]
		if (z != zDIM - 1)
			if ((g_data[x + y + z + zDIM ] & maxBits) == 1)
				++numlivecells;

		// [1,0]
		if (x != xDIM - 1  && z != zDIM - 1)
			if ((g_data[x + xDIM + y + z + zDIM ] & maxBits) == 1)
				++numlivecells;

		// [-1,1]
		if (x != 0 && y != xDIM -1 && z != zDIM - 1 )
			if ((g_data[x - xDIM + y + 1 + z + zDIM ] & maxBits) == 1)
				++numlivecells;

		// [0,1]
		if (y != xDIM -1 && z != zDIM - 1 )
			if ((g_data[x + y + 1 + z + zDIM ] & maxBits) == 1)
				++numlivecells;

		// [1,1]
		if (x != xDIM -1 && y != xDIM - 1 && z != zDIM - 1 )
			if ((g_data[x + xDIM + y + 1 + z + zDIM ] & maxBits) == 1)
				++numlivecells;

		return numlivecells;
	}

};