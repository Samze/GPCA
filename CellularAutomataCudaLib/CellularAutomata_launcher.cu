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

#include "CellularAutomata_kernal.cu"

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vector_types.h>
#include "Abstract2DCA.h"
#include "Abstract3DCA.h"

template<typename CAFunction>
extern float CUDATimeStep(CAFunction *func) {

	unsigned int *dev_pFlatGrid; //Pointers to device allocated memory
	int *dev_born; //to bornNo
	int *dev_survive; //to surviveNo
	CAFunction *dev_func;
	Abstract2DCA *dev_lattice;

	int* tempBorn;
	int* tempSurv;
	Abstract2DCA *tempLattice;
	unsigned int* tempGrid;

	cudaEvent_t start,stop; //Events for timings

	//START: Record duration of GPGPU processing
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start,0);

	int DIM = func->lattice->DIM;

	size_t noCells = DIM * DIM * sizeof(unsigned int);

	//Might need to flatten the 2d array ormaybe try "int2" type
	
	//TODO fix this name
	size_t size = sizeof(CAFunction);
	size_t sizeLattice = sizeof(Abstract2DCA);//func->lattice2->size();
	//Allocate suitable size memory on device
	cudaMalloc((void**) &dev_pFlatGrid, noCells);
	cudaMalloc((void**) &dev_func, size);
	cudaMalloc((void**) &dev_lattice, sizeLattice);

	cudaMalloc((void**) &dev_born, sizeof(int) * func->bornSize);
	cudaMalloc((void**) &dev_survive, sizeof(int) * func->surviveSize);

	//Make our 2D grid of blocks & threads (DIM/No of threads)
	//One pixel is one thread.
	dim3 blocks (DIM/20,DIM/20);
	dim3 threads(20,20);


	//copy our two dynamic arrays 
	cudaMemcpy(dev_born, func->bornNo, sizeof(int) * func->bornSize,
		cudaMemcpyHostToDevice);
	cudaMemcpy(dev_survive, func->surviveNo, sizeof(int) * func->surviveSize,
		cudaMemcpyHostToDevice);
	cudaMemcpy(dev_pFlatGrid, func->lattice->pFlatGrid, noCells,
		cudaMemcpyHostToDevice);
	
	tempGrid = func->lattice->pFlatGrid;

	func->lattice->pFlatGrid = dev_pFlatGrid;

	cudaMemcpy(dev_lattice, func->lattice, sizeLattice,
		cudaMemcpyHostToDevice);

	//We want to temporarily hold our pointers so we can reassign them after the object copy...
	tempBorn = func->bornNo;
	tempSurv = func->surviveNo;
	tempLattice = func->lattice;

	//reassign our pointers so we know where we put our dynamic arrays
	func->surviveNo = dev_survive;
	func->bornNo = dev_born;
	func->lattice = dev_lattice;

	
	//Copy our memory from Host to Device
	cudaMemcpy(dev_func, func,size,
		cudaMemcpyHostToDevice);

	kernal<<<blocks,threads>>>(dev_func);

	//Copy back to host
	cudaMemcpy(tempGrid, dev_pFlatGrid, noCells,
		cudaMemcpyDeviceToHost);

	//Reassign our dynamic array pointers
	func->surviveNo = tempSurv;
	func->bornNo = tempBorn;
	func->lattice = tempLattice;
	func->lattice->pFlatGrid = tempGrid;

	//STOP : processing done
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);

	float elapsedTime = 0;
	cudaEventElapsedTime(&elapsedTime, start, stop);

	
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	//fix up states - normalize
	for (int i = 0; i < DIM; ++i) {
		for (int j = 0; j < DIM; ++j) {
				func->lattice->pFlatGrid[i * DIM +j] = func->lattice->pFlatGrid[i * DIM +j] >> func->lattice->getNoBits();
		}
	}

	//Free memory on Device
	cudaFree(dev_pFlatGrid);
	cudaFree(dev_born);
	cudaFree(dev_survive);
	cudaFree(dev_func);

	return elapsedTime;
}

template<typename CAFunction>
extern float CUDATimeStep3D(CAFunction *func) {

	unsigned int *dev_pFlatGrid; //Pointers to device allocated memory
	int *dev_born; //to bornNo
	int *dev_survive; //to surviveNo
	unsigned int* dev_neighCount;
	CAFunction *dev_func;
	Abstract3DCA *dev_lattice;

	int* tempBorn;
	int* tempSurv;
	unsigned int* tempNeigh;
	Abstract3DCA *tempLattice;
	unsigned int* tempGrid;

	cudaEvent_t start,stop; //Events for timings

	//START: Record duration of GPGPU processing
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start,0);
	
	int DIM = func->lattice->DIM;

	size_t noCells = DIM * DIM * DIM * sizeof(unsigned int);
	//Might need to flatten the 2d array ormaybe try "int2" type
	
	//TODO fix this name
	size_t size = sizeof(CAFunction);

	//TODO Add this 
	//size_t sizeLattice = func->lattice->size();
	size_t sizeLattice = sizeof(Abstract3DCA);

	//Allocate suitable size memory on device
	cudaMalloc((void**) &dev_pFlatGrid, noCells);
	cudaMalloc((void**) &dev_func, size);
	cudaMalloc((void**) &dev_lattice, sizeLattice);

	cudaMalloc((void**) &dev_born, sizeof(int) * func->bornSize);
	cudaMalloc((void**) &dev_survive, sizeof(int) * func->surviveSize);
	cudaMalloc((void**) &dev_neighCount, noCells);


	//Make our 3D grid of blocks & threads (DIM/No of threads)
	//One pixel is one thread.
	/*dim3 blocks (1,1,1);
	dim3 threads(8,8,8);*/
	
	dim3 threads(16,16);
	dim3 blocks (DIM/threads.x + 1,(DIM/threads.y + 1) * DIM);

	//copy our two dynamic arrays 
	cudaMemcpy(dev_born, func->bornNo, sizeof(int) * func->bornSize,
		cudaMemcpyHostToDevice);
	cudaMemcpy(dev_survive, func->surviveNo, sizeof(int) * func->surviveSize,
		cudaMemcpyHostToDevice);
	cudaMemcpy(dev_pFlatGrid, func->lattice->pFlatGrid, noCells,
		cudaMemcpyHostToDevice);
	
	tempGrid = func->lattice->pFlatGrid;
	tempNeigh = func->lattice->neighbourCount;

	func->lattice->pFlatGrid = dev_pFlatGrid;
	func->lattice->neighbourCount = dev_neighCount;

	cudaMemcpy(dev_lattice, func->lattice, sizeLattice,
		cudaMemcpyHostToDevice);

	//We want to temporarily hold our pointers so we can reassign them after the object copy...
	tempBorn = func->bornNo;
	tempSurv = func->surviveNo;
	tempLattice = func->lattice;

	//reassign our pointers so we know where we put our dynamic arrays
	func->surviveNo = dev_survive;
	func->bornNo = dev_born;
	func->lattice = dev_lattice;
	
	//Copy our memory from Host to Device
	cudaMemcpy(dev_func, func,size,
		cudaMemcpyHostToDevice);

	kernal3DTest<<<blocks,threads>>>(dev_func);

	//Copy back to host
	cudaMemcpy(tempGrid, dev_pFlatGrid, noCells,
		cudaMemcpyDeviceToHost);

	//Because of our func currently holding a device pointer, we need to use a
	//temp pointer.
	cudaMemcpy(tempNeigh, dev_neighCount, noCells,
		cudaMemcpyDeviceToHost);


	//Reassign our dynamic array pointers
	func->surviveNo = tempSurv;
	func->bornNo = tempBorn;

	func->lattice = tempLattice;
	func->lattice->pFlatGrid = tempGrid;
	func->lattice->neighbourCount = tempNeigh;

	//STOP : processing done
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);

	float elapsedTime = 0;
	cudaEventElapsedTime(&elapsedTime, start, stop);

	
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	//fix up states - normalize, this could be another kernal really..

	for (int i = 0; i < DIM * DIM; ++i) {
		for (int j = 0; j < DIM; ++j) {
				func->lattice->pFlatGrid[i * DIM +j] = func->lattice->pFlatGrid[i * DIM +j] >> func->lattice->getNoBits();
		}
	}


	//Free memory on Device
	cudaFree(dev_pFlatGrid);
	cudaFree(dev_born);
	cudaFree(dev_survive);
	cudaFree(dev_func);
	cudaFree(dev_neighCount);

	return elapsedTime;
}

//TODO add support for this.
//const char* errorCheck() {
//	cudaError_t err = cudaGetLastError();
//	if (err != cudaSuccess) {
//		return cudaGetErrorString(err);
//	}
//	return  NULL;
//}
