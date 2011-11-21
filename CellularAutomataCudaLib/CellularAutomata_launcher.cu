#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vector_types.h>
#include "CellularAutomata_kernal.cu"


template<typename CAFunction>
extern float CUDATimeStep(int* pFlatGrid, int DIM, CAFunction *func) {

	int *dev_pFlatGrid; //Pointers to device allocated memory
	int *dev_DIM;
	int *dev_born; //to bornNo
	int *dev_survive; //to surviveNo
	CAFunction *dev_func;

	cudaEvent_t start,stop; //Events for timings

	//START: Record duration of GPGPU processing
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start,0);

	size_t noCells = DIM * DIM * sizeof(int);
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
	cudaMemcpy(dev_survive, func->surviveNo, sizeof(int) * func->surviveSize,
		cudaMemcpyHostToDevice);
	cudaMemcpy(dev_born, func->bornNo, sizeof(int) * func->bornSize,
		cudaMemcpyHostToDevice);

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
	cudaFree(dev_func);
	cudaFree(dev_born);
	cudaFree(dev_survive);

	return elapsedTime;
}

