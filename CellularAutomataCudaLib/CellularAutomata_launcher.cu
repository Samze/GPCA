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

#include "CellularAutomata_launcher.h"

#include "CellularAutomata_kernal.cu"

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vector_types.h>
#include "Lattice2D.h"
#include "Lattice3D.h"
#include <vector>

//TEMP REMOVE LATER
#include "SCIARA.h"

template<typename CAFunction>
extern float CUDATimeStepSCIARA(CAFunction *func) {

	CAFunction *dev_func;
	Lattice2D *dev_lattice;
	Lattice2D *tempLattice;

	cudaEvent_t start,stop; //Events for timings

	//START: Record duration of GPGPU processing
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start,0);

	int xDIM = func->getLattice()->getXSize();
	int yDIM = func->getLattice()->yDIM;

	//Might need to flatten the 2d array ormaybe try "int2" type

	//TODO fix this name
	size_t size = sizeof(CAFunction);
	size_t sizeLattice = func->lattice->size();
	//Allocate suitable size memory on device
	map<void**, size_t>* hostDynamicMap = func->getDynamicArrays();
	map<void**, size_t>::const_iterator iter;
	map<void**,void*> tempPointers;

	for(iter = hostDynamicMap->begin(); iter != hostDynamicMap->end(); ++iter) {

		void** tempPointer = (*iter).first;
		void* dataPointer = *(*iter).first;
		tempPointers.insert(make_pair(tempPointer,dataPointer));
	}

	vector<void*>* devicePointers = setupDynamicArrays(*hostDynamicMap);


	cudaMalloc((void**) &dev_func, size);
	cudaMalloc((void**) &dev_lattice, sizeLattice);

	//Make our 2D grid of blocks & threads (DIM/No of threads)
	//One pixel is one thread.
	dim3 threads(16,16);
	dim3 blocks (xDIM/threads.x + 1,(yDIM/threads.y + 1));

	//dim3 threads(21,21); //They are +2 for shared memory padding!
	//dim3 blocks (xDIM/20,yDIM/20);

	int count = 0;

	map<void**, void*>::const_iterator iterTP;
	
	for(iterTP = tempPointers.begin(); iterTP != tempPointers.end(); ++iterTP) {

		void** tmpPointer = (*iterTP).first;

		*tmpPointer = devicePointers->at(count);
		++count;
	}

	cudaMemcpy(dev_lattice, func->getLattice(), sizeLattice,
		cudaMemcpyHostToDevice);

	//We want to temporarily hold our pointers so we can reassign them after the object copy...


	tempLattice = func->getLattice();
	//reassign our pointers so we know where we put our dynamic arrays
	func->lattice = dev_lattice;

	
	//Copy our memory from Host to Device
	cudaMemcpy(dev_func, func,size,
		cudaMemcpyHostToDevice);

	kernal<<<blocks,threads>>>(dev_func);

	//Reassign our dynamic pointers
	count = 0;

	for(iterTP = tempPointers.begin(); iterTP != tempPointers.end(); ++iterTP) {

		void** pointerLoc = (*iterTP).first;
		void* tmpPointer = (*iterTP).second;
		
		size_t size = hostDynamicMap->at(pointerLoc);

		cudaMemcpy(tmpPointer, devicePointers->at(count), size,
			cudaMemcpyDeviceToHost);

		*pointerLoc = tmpPointer;
		//Copy back to host


		//pointerLoc = &tmpPointer;

		cudaFree(devicePointers->at(count));

		++count;
	}
	
	//Delete our lists
	delete hostDynamicMap;
	delete devicePointers;

	//Reassign our dynamic array pointers
	func->lattice = tempLattice;

	//STOP : processing done
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);

	float elapsedTime = 0;
	cudaEventElapsedTime(&elapsedTime, start, stop);

	
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	//fix up states - normalize
	//for (int i = 0; i < DIM; ++i) {
	//	for (int j = 0; j < DIM; ++j) {
	//			func->lattice->pFlatGrid[i * DIM +j] = func->lattice->pFlatGrid[i * DIM +j] >> func->lattice->getNoBits();
	//	}
	//}

	//Free memory on Device
	cudaFree(dev_func);

	return elapsedTime;
}


template<typename CAFunction>
extern float CUDATimeStep(CAFunction *func) {

	CAFunction *dev_func;
	Lattice2D *dev_lattice;

	Lattice2D *tempLattice;

	cudaEvent_t start,stop; //Events for timings

	//START: Record duration of GPGPU processing
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start,0);

	int xDIM = func->lattice->getXSize();
	int yDIM = func->lattice->yDIM;
	

	//TODO fix this name
	size_t size = sizeof(CAFunction);
	size_t sizeLattice = func->lattice->size();

	//Allocate suitable size memory on device
	map<void**, size_t>* hostDynamicMap = func->getDynamicArrays();
	map<void**, size_t>::const_iterator iter;
	map<void**,void*> tempPointers;

	for(iter = hostDynamicMap->begin(); iter != hostDynamicMap->end(); ++iter) {

		void** tempPointer = (*iter).first;
		void* dataPointer = *(*iter).first;
		tempPointers.insert(make_pair(tempPointer,dataPointer));
	}

	vector<void*>* devicePointers = setupDynamicArrays(*hostDynamicMap);

	cudaMalloc((void**) &dev_func, size);
	cudaMalloc((void**) &dev_lattice, sizeLattice);


	//Make our 2D grid of blocks & threads (DIM/No of threads)
	//One pixel is one thread.6
	dim3 threads(22,22); //They are +2 for shared memory padding!
	dim3 blocks (xDIM/20 + 1,yDIM/20 + 1);


	//dim3 threads(16,16);
	//dim3 blocks ((xDIM/threads.x) + 1,(yDIM/threads.y) + 1);

	//We want to temporarily hold our pointers so we can reassign them after the object copy...
	int count = 0;

	map<void**, void*>::const_iterator iterTP;
	
	for(iterTP = tempPointers.begin(); iterTP != tempPointers.end(); ++iterTP) {

		void** tmpPointer = (*iterTP).first;

		*tmpPointer = devicePointers->at(count);
		++count;
	}

	//copy our two dynamic arrays 
	cudaMemcpy(dev_lattice, func->getLattice(), sizeLattice,
		cudaMemcpyHostToDevice);

	tempLattice = func->getLattice();
	func->lattice = dev_lattice;

	
	//Copy our memory from Host to Device
	cudaMemcpy(dev_func, func,size,
		cudaMemcpyHostToDevice);

	kernalSharedMem<<<blocks,threads>>>(dev_func);

	//Reassign our dynamic array pointers
	count = 0;

	for(iterTP = tempPointers.begin(); iterTP != tempPointers.end(); ++iterTP) {

		void** pointerLoc = (*iterTP).first;
		void* tmpPointer = (*iterTP).second;
		
		size_t size = hostDynamicMap->at(pointerLoc);

		cudaMemcpy(tmpPointer, devicePointers->at(count), size,
			cudaMemcpyDeviceToHost);

		*pointerLoc = tmpPointer;
		//Copy back to host


		//pointerLoc = &tmpPointer;

		cudaFree(devicePointers->at(count));

		++count;
	}

	//Delete our lists
	delete hostDynamicMap;
	delete devicePointers;

	func->lattice = tempLattice;

	//STOP : processing done
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);

	float elapsedTime = 0;
	cudaEventElapsedTime(&elapsedTime, start, stop);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	unsigned int* newGrid = (unsigned int*) func->getLattice()->pFlatGrid;



	//fix up states - normalize
	for (int i = 0; i < xDIM; ++i) {
		for (int j = 0; j < yDIM; ++j) {
				newGrid[i * xDIM +j] = newGrid[i * xDIM + j] >> func->getLattice()->getNoBits();
		}
	}

	//Free memory on Device
	cudaFree(dev_func);

	return elapsedTime;
}

template<typename CAFunction>
extern float CUDATimeStep3D(CAFunction *func) {

	//unsigned int *dev_pFlatGrid; //Pointers to device allocated memory
	unsigned int* dev_neighCount;
	CAFunction *dev_func;
	Lattice3D *dev_lattice;

	unsigned int* tempNeigh;
	Lattice3D *tempLattice;

	cudaEvent_t start,stop; //Events for timings

	//START: Record duration of GPGPU processing
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start,0);
	
	int xDIM = func->getLattice()->getXSize();
	int yDIM = func->getLattice()->yDIM;
	int zDIM = func->getLattice()->zDIM;

	size_t noCells = xDIM * yDIM * zDIM * sizeof(unsigned int);
	//Might need to flatten the 2d array ormaybe try "int2" type
	
	//TODO fix this name
	size_t size = sizeof(CAFunction);

	map<void**, size_t>* hostDynamicMap = func->getDynamicArrays();

	map<void**, size_t>::const_iterator iter;
	
	map<void**,void*> tempPointers;

	for(iter = hostDynamicMap->begin(); iter != hostDynamicMap->end(); ++iter) {

		void** tempPointer = (*iter).first;
		void* dataPointer = *(*iter).first;
		tempPointers.insert(make_pair(tempPointer,dataPointer));
	}

	vector<void*>* devicePointers = setupDynamicArrays(*hostDynamicMap);

	//TODO Add this 
	//size_t sizeLattice = func->lattice->size();
	size_t sizeLattice = func->lattice->size();


	//Allocate suitable size memory on device
//	cudaMalloc((void**) &dev_pFlatGrid, noCells);
	cudaMalloc((void**) &dev_func, size);
	cudaMalloc((void**) &dev_lattice, sizeLattice);

	//cudaMalloc((void**) &dev_born, sizeof(int) * func->bornSize);
	//cudaMalloc((void**) &dev_survive, sizeof(int) * func->surviveSize);
	cudaMalloc((void**) &dev_neighCount, noCells);


	//Do our specific setup, such as copying any dynamic arrays we may require.
	//func->setup();

	//Make our 3D grid of blocks & threads (DIM/No of threads)
	//One pixel is one thread

	//dim3 threads(8,8,8);
	
	//dim3 blocks((xDIM/(threads.x - 2) + 1) * (zDIM/(threads.z- 2) + 1),yDIM/(threads.y - 2) + 1);

	dim3 threads(16,16);
	dim3 blocks (xDIM/threads.x + 1,(xDIM/threads.y + 1) * xDIM);

	//copy our two dynamic arrays 
	//cudaMemcpy(dev_born, func->bornNo, sizeof(int) * func->bornSize,
	//	cudaMemcpyHostToDevice);
	//cudaMemcpy(dev_survive, func->surviveNo, sizeof(int) * func->surviveSize,
	//	cudaMemcpyHostToDevice);
	//cudaMemcpy(dev_pFlatGrid, func->lattice->pFlatGrid, noCells,
	//	cudaMemcpyHostToDevice);
	
	tempNeigh = func->getLattice()->neighbourCount;
/*
	func->lattice->pFlatGrid = dev_pFlatGrid;*/
	func->getLattice()->neighbourCount = dev_neighCount;

	//reassign our pointers so we know where we put our dynamic arrays
	int count = 0;

	map<void**, void*>::const_iterator iterTP;
	
	for(iterTP = tempPointers.begin(); iterTP != tempPointers.end(); ++iterTP) {

		void** tmpPointer = (*iterTP).first;

		*tmpPointer = devicePointers->at(count);
		++count;
	}

	cudaMemcpy(dev_lattice, func->getLattice(), sizeLattice,
		cudaMemcpyHostToDevice);
	
	tempLattice = func->getLattice();
	func->lattice = dev_lattice;
	//func->surviveNo = dev_survive;
	//func->bornNo = dev_born;

	//We want to temporarily hold our pointers so we can reassign them after the object copy...
	//tempBorn = func->bornNo;
	//tempSurv = func->surviveNo;


	//Copy our memory from Host to Device
	cudaMemcpy(dev_func, func,size,
		cudaMemcpyHostToDevice);

	kernal3D<<<blocks,threads>>>(dev_func);


	//Because of our func currently holding a device pointer, we need to use a
	//temp pointer.
	cudaMemcpy(tempNeigh, dev_neighCount, noCells,
		cudaMemcpyDeviceToHost);

	//Reassign our dynamic pointers
	count = 0;

	for(iterTP = tempPointers.begin(); iterTP != tempPointers.end(); ++iterTP) {

		void** pointerLoc = (*iterTP).first;
		void* tmpPointer = (*iterTP).second;
		
		size_t size = hostDynamicMap->at(pointerLoc);

		cudaMemcpy(tmpPointer, devicePointers->at(count), size,
			cudaMemcpyDeviceToHost);

		*pointerLoc = tmpPointer;
		//Copy back to host


		//pointerLoc = &tmpPointer;

		cudaFree(devicePointers->at(count));

		++count;
	}
	
	//Delete our lists
	delete hostDynamicMap;
	delete devicePointers;

	//func->surviveNo = tempSurv;
	//func->bornNo = tempBorn;
	
	func->lattice = tempLattice;
	//func->lattice->pFlatGrid = tempGrid;
	func->getLattice()->neighbourCount = tempNeigh;

	//STOP : processing done
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);

	float elapsedTime = 0;
	cudaEventElapsedTime(&elapsedTime, start, stop);

	
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	//fix up states - normalize, this could be another kernal really..

	unsigned int* newGrid = (unsigned int*)func->getLattice()->getGrid();

	for (int i = 0; i < xDIM; ++i) {
		for (int j = 0; j < yDIM ; ++j) {
			for (int k = 0; k < zDIM; ++k) {
				newGrid[(i * xDIM) + (k * zDIM * zDIM) +j] = newGrid[(i * xDIM) + (k * zDIM * zDIM) +j] >> func->getLattice()->getNoBits();
			}
		}
	}

	//Free memory on Device
	cudaFree(dev_func);
	cudaFree(dev_neighCount);

	return elapsedTime;
}

//template<typename CAFunction>
//extern float CUDATimeStep3DOLD(CAFunction *func) {
//
//	//unsigned int *dev_pFlatGrid; //Pointers to device allocated memory
//	unsigned int* dev_neighCount;
//	CAFunction *dev_func;
//	Lattice3D *dev_lattice;
//
//	unsigned int* tempNeigh;
//	Lattice3D *tempLattice;
//
//	cudaEvent_t start,stop; //Events for timings
//
//	//START: Record duration of GPGPU processing
//	cudaEventCreate(&start);
//	cudaEventCreate(&stop);
//
//	cudaEventRecord(start,0);
//	
//	int DIM = func->lattice->DIM;
//
//	size_t noCells = DIM * DIM * DIM * sizeof(unsigned int);
//	//Might need to flatten the 2d array ormaybe try "int2" type
//	
//	//TODO fix this name
//	size_t size = sizeof(CAFunction);
//
//	map<void**, size_t>* hostDynamicMap = func->getDynamicArrays();
//
//	map<void**, size_t>::const_iterator iter;
//	
//	map<void**,void*> tempPointers;
//
//	for(iter = hostDynamicMap->begin(); iter != hostDynamicMap->end(); ++iter) {
//
//		void** tempPointer = (*iter).first;
//		void* dataPointer = *(*iter).first;
//		tempPointers.insert(make_pair(tempPointer,dataPointer));
//	}
//
//	vector<void*>* devicePointers = setupDynamicArrays(*hostDynamicMap);
//
//	//TODO Add this 
//	//size_t sizeLattice = func->lattice->size();
//	size_t sizeLattice = sizeof(Abstract3DCA);
//
//
//	//Allocate suitable size memory on device
////	cudaMalloc((void**) &dev_pFlatGrid, noCells);
//	cudaMalloc((void**) &dev_func, size);
//	cudaMalloc((void**) &dev_lattice, sizeLattice);
//
//	//cudaMalloc((void**) &dev_born, sizeof(int) * func->bornSize);
//	//cudaMalloc((void**) &dev_survive, sizeof(int) * func->surviveSize);
//	cudaMalloc((void**) &dev_neighCount, noCells);
//
//
//	//Do our specific setup, such as copying any dynamic arrays we may require.
//	//func->setup();
//
//	//Make our 3D grid of blocks & threads (DIM/No of threads)
//	//One pixel is one thread.
//	/*dim3 blocks (1,1,1);
//	dim3 threads(8,8,8);*/
//	
//	dim3 threads(16,16);
//	dim3 blocks (DIM/threads.x + 1,(DIM/threads.y + 1) * DIM);
//
//	//copy our two dynamic arrays 
//	//cudaMemcpy(dev_born, func->bornNo, sizeof(int) * func->bornSize,
//	//	cudaMemcpyHostToDevice);
//	//cudaMemcpy(dev_survive, func->surviveNo, sizeof(int) * func->surviveSize,
//	//	cudaMemcpyHostToDevice);
//	//cudaMemcpy(dev_pFlatGrid, func->lattice->pFlatGrid, noCells,
//	//	cudaMemcpyHostToDevice);
//	
//	tempNeigh = func->lattice->neighbourCount;
///*
//	func->lattice->pFlatGrid = dev_pFlatGrid;*/
//	func->lattice->neighbourCount = dev_neighCount;
//
//	//reassign our pointers so we know where we put our dynamic arrays
//	int count = 0;
//
//	map<void**, void*>::const_iterator iterTP;
//	
//	for(iterTP = tempPointers.begin(); iterTP != tempPointers.end(); ++iterTP) {
//
//		void** tmpPointer = (*iterTP).first;
//
//		*tmpPointer = devicePointers->at(count);
//		++count;
//	}
//
//	cudaMemcpy(dev_lattice, func->lattice, sizeLattice,
//		cudaMemcpyHostToDevice);
//	
//	tempLattice = func->lattice;
//	func->lattice = dev_lattice;
//	//func->surviveNo = dev_survive;
//	//func->bornNo = dev_born;
//
//	//We want to temporarily hold our pointers so we can reassign them after the object copy...
//	//tempBorn = func->bornNo;
//	//tempSurv = func->surviveNo;
//
//
//	//Copy our memory from Host to Device
//	cudaMemcpy(dev_func, func,size,
//		cudaMemcpyHostToDevice);
//
//	kernal3D<<<blocks,threads>>>(dev_func);
//
//
//
//	//Because of our func currently holding a device pointer, we need to use a
//	//temp pointer.
//	cudaMemcpy(tempNeigh, dev_neighCount, noCells,
//		cudaMemcpyDeviceToHost);
//
//
//	//Reassign our dynamic pointers
//	count = 0;
//
//	for(iterTP = tempPointers.begin(); iterTP != tempPointers.end(); ++iterTP) {
//
//		void** pointerLoc = (*iterTP).first;
//		void* tmpPointer = (*iterTP).second;
//		
//		size_t size = hostDynamicMap->at(pointerLoc);
//
//		cudaMemcpy(tmpPointer, devicePointers->at(count), size,
//			cudaMemcpyDeviceToHost);
//
//		*pointerLoc = tmpPointer;
//		//Copy back to host
//
//
//		//pointerLoc = &tmpPointer;
//
//		cudaFree(devicePointers->at(count));
//
//		++count;
//	}
//	
//	//func->surviveNo = tempSurv;
//	//func->bornNo = tempBorn;
//
//	func->lattice = tempLattice;
//	//func->lattice->pFlatGrid = tempGrid;
//	func->lattice->neighbourCount = tempNeigh;
//
//	//STOP : processing done
//	cudaEventRecord(stop,0);
//	cudaEventSynchronize(stop);
//
//	float elapsedTime = 0;
//	cudaEventElapsedTime(&elapsedTime, start, stop);
//
//	
//	cudaEventDestroy(start);
//	cudaEventDestroy(stop);
//
//	//fix up states - normalize, this could be another kernal really..
//
//	unsigned int* newGrid = (unsigned int*)func->lattice->pFlatGrid;
//
//	for (int i = 0; i < DIM * DIM; ++i) {
//		for (int j = 0; j < DIM; ++j) {
//				//newGrid[i * DIM +j] = newGrid[i * DIM +j] >> func->lattice->getNoBits();
//		}
//	}
//
//	//Free memory on Device
//	cudaFree(dev_func);
//	cudaFree(dev_neighCount);
//
//	return elapsedTime;
//}
