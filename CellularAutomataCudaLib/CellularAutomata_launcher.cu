#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vector_types.h>
#include "CellularAutomata_kernal.cu"


template<typename CAFunction>
extern float CUDATimeStep(unsigned int* pFlatGrid, int DIM, CAFunction *func) {

	unsigned int *dev_pFlatGrid; //Pointers to device allocated memory
	int *dev_DIM;
	int *dev_born; //to bornNo
	int *dev_survive; //to surviveNo
	CAFunction *dev_func;

	int* tempBorn;
	int* tempSurv;

	cudaEvent_t start,stop; //Events for timings

	//START: Record duration of GPGPU processing
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start,0);

	size_t noCells = DIM * DIM * sizeof(unsigned int);
	//Might need to flatten the 2d array ormaybe try "int2" type
	
	//TODO fix this name
	size_t size = sizeof(CAFunction);
	//Allocate suitable size memory on device
	cudaMalloc((void**) &dev_pFlatGrid, noCells);
	cudaMalloc((void**) &dev_DIM, sizeof(int));
	cudaMalloc((void**) &dev_func, sizeof(CAFunction));

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

	//We want to temporarily hold our pointers so we can reassign them after the object copy...
	tempBorn = func->bornNo;
	tempSurv = func->surviveNo;

	//reassign our pointers so we know where we put our dynamic arrays
	func->surviveNo = dev_survive;
	func->bornNo = dev_born;
	
	
	
	//Copy our memory from Host to Device
	cudaMemcpy(dev_pFlatGrid, pFlatGrid, noCells,
		cudaMemcpyHostToDevice);
	cudaMemcpy(dev_DIM, &DIM, sizeof(int),
		cudaMemcpyHostToDevice);
	cudaMemcpy(dev_func, func, sizeof(CAFunction),
		cudaMemcpyHostToDevice);


	kernal<<<blocks,threads>>>(dev_pFlatGrid, dev_DIM, dev_func);

	//Copy back to host
	cudaMemcpy(pFlatGrid, dev_pFlatGrid, noCells,
		cudaMemcpyDeviceToHost);

	//Reassign our dynamic array pointers
	func->surviveNo = tempSurv;
	func->bornNo = tempBorn;

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
				pFlatGrid[i * DIM +j] = pFlatGrid[i * DIM +j] >> func->noBits;
		}
	}


	//Free memory on Device
	cudaFree(dev_pFlatGrid);
	cudaFree(dev_DIM);
	cudaFree(dev_born);
	cudaFree(dev_survive);
	cudaFree(dev_func);

	return elapsedTime;
}

template<typename CAFunction>
extern float CUDATimeStep3D(unsigned int* pFlatGrid, int DIM, CAFunction *func) {

	unsigned int *dev_pFlatGrid; //Pointers to device allocated memory
	int *dev_DIM;
	int *dev_born; //to bornNo
	int *dev_survive; //to surviveNo
	unsigned int* dev_neighCount;
	CAFunction *dev_func;

	int* tempBorn;
	int* tempSurv;
	unsigned int* tempNeigh;

	cudaEvent_t start,stop; //Events for timings

	//START: Record duration of GPGPU processing
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start,0);

	size_t noCells = DIM * DIM * DIM * sizeof(unsigned int);
	//Might need to flatten the 2d array ormaybe try "int2" type
	
	//TODO fix this name
	size_t size = sizeof(CAFunction);
	//Allocate suitable size memory on device
	cudaMalloc((void**) &dev_pFlatGrid, noCells);
	cudaMalloc((void**) &dev_DIM, sizeof(int));
	cudaMalloc((void**) &dev_func, sizeof(CAFunction));

	cudaMalloc((void**) &dev_born, sizeof(int) * func->bornSize);
	cudaMalloc((void**) &dev_survive, sizeof(int) * func->surviveSize);
	cudaMalloc((void**) &dev_neighCount, noCells);


	//Make our 3D grid of blocks & threads (DIM/No of threads)
	//One pixel is one thread.
	/*dim3 blocks (1,1,1);
	dim3 threads(8,8,8);*/
	
	dim3 threads(8,8);
	dim3 blocks (DIM/threads.x + 1,(DIM/threads.y + 1) * DIM);

	//copy our two dynamic arrays 
	cudaMemcpy(dev_born, func->bornNo, sizeof(int) * func->bornSize,
		cudaMemcpyHostToDevice);
	cudaMemcpy(dev_survive, func->surviveNo, sizeof(int) * func->surviveSize,
		cudaMemcpyHostToDevice);

	cudaMemcpy(dev_neighCount, func->neighbourCount, noCells,
		cudaMemcpyHostToDevice);

	//We want to temporarily hold our pointers so we can reassign them after the object copy...
	tempBorn = func->bornNo;
	tempSurv = func->surviveNo;
	tempNeigh = func->neighbourCount;

	//reassign our pointers so we know where we put our dynamic arrays
	func->surviveNo = dev_survive;
	func->bornNo = dev_born;
	func->neighbourCount = dev_neighCount;
	
	
	
	//Copy our memory from Host to Device
	cudaMemcpy(dev_pFlatGrid, pFlatGrid, noCells,
		cudaMemcpyHostToDevice);
	cudaMemcpy(dev_DIM, &DIM, sizeof(int),
		cudaMemcpyHostToDevice);
	cudaMemcpy(dev_func, func, sizeof(CAFunction),
		cudaMemcpyHostToDevice);

	//TODO memory leak?
	//delete[] pFlatGrid;

	kernal3DTest<<<blocks,threads>>>(dev_pFlatGrid, dev_DIM, dev_func);

	//Copy back to host
	cudaMemcpy(pFlatGrid, dev_pFlatGrid, noCells,
		cudaMemcpyDeviceToHost);

	//Because of our func currently holding a device pointer, we need to use a
	//temp pointer.
	cudaMemcpy(tempNeigh, dev_neighCount, noCells,
		cudaMemcpyDeviceToHost);


	//Reassign our dynamic array pointers
	func->surviveNo = tempSurv;
	func->bornNo = tempBorn;
	func->neighbourCount = tempNeigh;

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
				pFlatGrid[i * DIM +j] = pFlatGrid[i * DIM +j] >> func->noBits;
		}
	}


	//Free memory on Device
	cudaFree(dev_pFlatGrid);
	cudaFree(dev_DIM);
	cudaFree(dev_born);
	cudaFree(dev_survive);
	cudaFree(dev_func);
	cudaFree(dev_neighCount);

	return elapsedTime;
}
