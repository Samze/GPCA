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

#ifndef ABSTRACT_LATTICE_H
#define ABSTRACT_LATTICE_H

#include "device_launch_parameters.h"

#define DLLExport __declspec(dllexport)

class AbstractLattice
{

public:
	DLLExport AbstractLattice(void) {}
	DLLExport AbstractLattice(unsigned int dim);
	DLLExport AbstractLattice(unsigned int dim, void* grid);

	DLLExport virtual ~AbstractLattice(void);
	
	__host__ virtual size_t size() const = 0;
	
	__host__ __device__ int getNoBits() { 
		return noBits;
	}

	__host__ __device__ unsigned int getXSize(){
		return xDIM;
	}
	__host__ __device__ unsigned int getMaxBits(){
		return maxBits;
	}

	DLLExport __host__ void setMaxBits(unsigned int);
	DLLExport __host__ void setNoBits(unsigned int);
	DLLExport __host__ void setNeighbourhoodType(int);
	DLLExport __host__ void setNoElements(unsigned int);
	DLLExport __host__ void setXSize(unsigned int);

	__host__ __device__ int getNeighbourhoodType(){
		return neighbourhoodType;
	}
	__host__ __device__ unsigned int getNoElements(){
		return noElements;
	}
	
	__host__ __device__ void* getGrid(){
		return pFlatGrid;
	}


	unsigned int xDIM;
	void *pFlatGrid;

protected:
	unsigned int noBits;
	unsigned int maxBits;
	int neighbourhoodType;
	unsigned int noElements;
	
	
	//__device__ __host__ virtual void getNeighbourhood(int* neighbourStates, unsigned int* g_data, int gLocation);

	//This next line should be here to provide 'proper' virtual inheritence, sadly it is only supported on CUDA sm_2x architecture.
	//__device__ __host__ int applyFunction(int*,int,int,int) {
	//	return 3;
	//}

};


#endif