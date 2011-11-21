#pragma once //TODO Get rid of these

#include "device_launch_parameters.h"
#include "AbstractCellularAutomata.h"


class Abstract2DCA : public AbstractCellularAutomata
{

public:
	DLLExport Abstract2DCA(void);
	DLLExport ~Abstract2DCA(void);
	
	int m_states;
	int noBits;
	
    int* surviveNo;
    int  surviveSize;

	int* bornNo;
    int bornSize;

	int neighbourhoodType;

	DLLExport void setStates(int);

	__device__ __host__ void setSurviveNo(int* list, int size) {
		surviveNo = list;
		surviveSize = size;
	}

	__device__ __host__ void setBornNo(int* list, int size) {
		bornNo = list;
		bornSize = size;
	}


	//Always better to use constants than defines (effective C++)
	static const int MOORE = 0;
	static const int VON_NEUMANN = 1;
	
	__device__ __host__ int getNeighbourhood(int* g_data, int x, int y, int xDIM, int neighbourhood) {

		switch(neighbourhood) {
		case MOORE:
			return getMooresNeighbourhood(g_data,x,y,xDIM);
		case VON_NEUMANN:
			return getVonNeumannNeighbourhood(g_data,x,y,xDIM);
		default:
			return 0;
		}
	}

	//probably a much better way to figure out the moores neighbourhood
	__device__ __host__ int getMooresNeighbourhood(int* g_data, int x, int y, int xDIM) {

		//get neighbours for cell x,y
		int numlivecells = 0;

		// [-1,-1]
		if (x != 0 && y != 0)
			if (g_data[x - xDIM + y - 1] & 1 == 1)
				++numlivecells;

		// [0,-1]
		if ( y != 0)
			if (g_data[x + y - 1] & 1 == 1)
				++numlivecells;

		// [1,-1]
		if (x != xDIM - 1 && y != 0 )
			if (g_data[x + xDIM + y - 1] & 1 == 1)
				++numlivecells;

		// [-1,0]
		if (x != 0)
			if (g_data[x - xDIM + y] & 1 == 1)
				++numlivecells;	

		// [1,0]
		if (x != xDIM - 1)
			if (g_data[x + xDIM + y] & 1 == 1)
				++numlivecells;

		// [-1,1]
		if (x != 0 && y != xDIM -1 )
			if (g_data[x - xDIM + y + 1] & 1 == 1)
				++numlivecells;

		// [0,1]
		if (y != xDIM -1 )
			if (g_data[x + y + 1] & 1 == 1)
				++numlivecells;

		// [1,1]
		if (x != xDIM -1 && y != xDIM -1 )
			if (g_data[x + xDIM + y + 1] & 1 == 1)
				++numlivecells;

		return numlivecells;
	}

		//probably a much better way to figure out the moores neighbourhood
	__device__ __host__  int getVonNeumannNeighbourhood(int* g_data, int x, int y, int xDIM) {

		//get neighbours for cell x,y
		int numlivecells = 0;

		// [0,-1]
		if ( y != 0)
			if (g_data[x + y - 1] & 1 == 1)
				++numlivecells;


		// [-1,0]
		if (x != 0)
			if (g_data[x - xDIM + y] & 1 == 1)
				++numlivecells;	

		// [1,0]
		if (x != xDIM - 1)
			if (g_data[x + xDIM + y] & 1 == 1)
				++numlivecells;

	
		// [0,1]
		if (y != xDIM -1 )
			if (g_data[x + y + 1] & 1 == 1)
				++numlivecells;

		return numlivecells;
	}
};

