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

/**
* An abstract class used for lattices. A Lattice in a Cellular Automata is the space in which the cells
* occupy. This can be in any number of dimensions, of any size. All lattice data is held in a 1Dimensional array. 
* Any dimension is addressible by knowing the size dimensions. For example x=5, y=5 in a 2D CA can be addressed in a 1D grid by idx = x * DIM + y.
* This class defines the size attribute to represent 1 dimension. Any new lattice dimension should inherit from this, adding in their own Subsequent dimensions.
* Lattice information stored is of type void*, meaning data can be represented as simple types. e.g. (ints) or more complex, such as structs.
* Both 2D and 3D are supported thus far.
*@see Lattice2D
*@see Lattice3D
*/
class AbstractLattice
{

public:
	/**
	* Default constructor. Dimensions must be setup before use.
	*/
	DLLExport AbstractLattice();

	/**
	*Constructor providing the allowed setting of the size or the first dimension.
	*@param xSize Integer size of the first dimensions.
	*/
	DLLExport AbstractLattice(unsigned int xSize);
	
	/**
	*Constructor providing the allowed setting of the size or the first dimension and the new grid.
	*@param xSize Integer size of the first dimensions.
	*@param grid A pointer to the grid location..
	*/
	DLLExport AbstractLattice(unsigned int xSize, void* grid);

	/**
	*Virtual constructor. This class is meant to be inherited from.
	*/
	DLLExport virtual ~AbstractLattice(void);
	
	/**
	* To allocate the appropriate amount of memory on the GPU. The lattice class must specify how big it is. This is the actual class itself.
	* ie. a typical implemented would be.
	* return sizeof(LatticeXD);
	*@return The size of the clas in bytes.
	*/
	__host__ virtual size_t size() const = 0;

	
	/**
	* Gets the first dimension size from the lattice. E.g for a one dimension CA, this would be the number of cells wide it is.
	*@returns The size of the first dimension.
	*/
	__host__ __device__ unsigned int getXSize(){
		return xDIM;
	}
	
	/**
	* Sets the first dimension size for the CA.
	* CA.
	* @param newXSize The size of the first dimension.
	*/
	DLLExport __host__ void setXSize(unsigned int newXSize);

	/**
	* Gets the number of bits that the number of states require. This method is only applicable if the states are integers. 
	* For example if the CA has 2 States, dead and alive, then the number of bits to store this information is 1.
	* This method is used internally and should not be required by a client.
	* @return The minimum number of bits required to store the states.
	*
	*/
	__host__ __device__ int getNoBits() { 
		return noBits;
	}

	/**
	* Gets the maximum value of the number of bits required for state data. Ie. 2 bits have a maximum value of 3. 
	* This method is used internally and should not be required by a client.
	* @return The maximum value for a the currently set NoBits.
	*/
	__host__ __device__ unsigned int getMaxBits(){
		return maxBits;
	}

	/**
	* Sets the no bits required for state data. This is only applicable when state data is an integer type. 
	* This method is used internally and should not be required by a client.
	*@param newNoBits The number of bits required to hold the state data.
	*@see getNoBits()
	*/
	DLLExport __host__ void setNoBits(unsigned int newNoBits);

	/**
	* Sets the maximum value of the number of bits required for state data,
	* This method is used internally and should not be required by a client.
	* @param newMaxBits The maximum value for a the currently set NoBits.
	*/
	DLLExport __host__ void setMaxBits(unsigned int newMaxBits);
	
	/**
	* Gets the neighbourhood type. The neighbourhood type defines the relationship of interaction between neighbouring cells in the CA..a
	* It is the number of cells it interacts to update it's state
	* It is recommended that subclasses enum this value.
	* @return  The neighbourtype.
	*/
	__host__ __device__ int getNeighbourhoodType(){
		return neighbourhoodType;
	}
	/**
	* Sets the neighbourhood type. The neighbourhood type defines the relationship of interaction between neighbouring cells in the CA..a
	* It is the number of cells it interacts to update it's state
	* It is recommended that subclasses enum this value.
	* @param newNeighbourType The new neighbourtype.
	* @see Lattice2D
	* @see Lattice3D
	*/
	DLLExport __host__ void setNeighbourhoodType(int newNeighbourType);

	/**
	* Sets the number of total elements in the Lattice. This value is a multiplication of all the dimension sizes. Ie. x-max * y-max, for a 2D
	* CA.
	* @param newNoElements The maximum number of elements in the CA.
	*/
	DLLExport __host__ void setNoElements(unsigned int newNoElements);

	/**
	* Gets the number of total elements in the Lattice. This value is a multiplication of all the dimension sizes. Ie. x-max * y-max, for a 2D
	* CA.
	* @return The maximum number of elements in the CA.
	*/
	__host__ __device__ unsigned int getNoElements(){
		return noElements;
	}
	
	/**
	* Returns the pointer to the void* array that contains the lattice information. Note this is a one dimensional representation version of
	* the lattice, regardless of it's dimension. All lattice data is stored in a 1D array.
	*
	*/
	__host__ __device__ void* getGrid(){
		return pFlatGrid;
	}


	unsigned int xDIM; /**< The first dimension. This variable is only public for __device__ access, hosts should use the getters and setters.*/
	void *pFlatGrid; /**< The lattice grid, a one dimensional version of the lattice data.. This variable is only public for __device__ access, hosts should use the getters and setters.*/

protected:
	unsigned int noBits; 
	unsigned int maxBits;
	int neighbourhoodType;
	unsigned int noElements;
	
	
	//This next line should be here to provide 'proper' virtual inheritence, sadly it is only supported on CUDA sm_2x architecture.
	//__device__ __host__ int applyFunction(int*,int,int,int) {
	//	return 3;
	//}

};


#endif