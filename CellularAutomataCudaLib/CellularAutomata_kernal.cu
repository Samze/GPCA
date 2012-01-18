//#include "cellularautomata_kernal_DLL.h"

template <typename CAFunction>
__global__ void kernal(unsigned int* g_data, int* DIM, CAFunction* func) {
	
	int x = threadIdx.x + blockIdx.x * blockDim.x; 
    int y = threadIdx.y + blockIdx.y * blockDim.y;

	if( !(x > *DIM) &&  !(y > *DIM)) {//Guard against launching too many threads
	//set new cell state.
	
		//__syncthreads();

		g_data[(x * *DIM) + y] = func->applyFunction(g_data,x,y,*DIM);
	
		//g_data[(x * *DIM) + y] = (x * *DIM) + y;
	}
}

template <typename CAFunction>
__global__ void kernal3DTest(unsigned int* g_data, int* DIM, CAFunction* func) {
	
	int x = threadIdx.x + blockIdx.x * blockDim.x; 
	
	int slice = *DIM/blockDim.y + 1;

	int y = (blockIdx.y % slice) * blockDim.y + threadIdx.y;
	int z = blockIdx.y/slice;


	if( x >= *DIM ||  y >= *DIM || z >= *DIM) //Guard against launching too many threads
		return;
	
	//__syncthreads();

	g_data[(z * *DIM * *DIM) + (x * *DIM) + y] = func->applyFunction(g_data,x,y,z,*DIM);
	
}

template <typename CAFunction>
__global__ void kernal3D(unsigned int* g_data, int* DIM, CAFunction* func) {
	
	int x = threadIdx.x + blockIdx.x * blockDim.x; 
    int y = threadIdx.y + blockIdx.y * blockDim.y;
	int z = threadIdx.z;// + blockIdx.z * blockDim.z;

	//This is our fake z area
	if( blockIdx.x >= gridDim.x/2) {
		//z = threadIdx.z + blockDim.z;
		x = threadIdx.x + (blockIdx.x - gridDim.x/2) * blockDim.x;
	}

	if( blockIdx.y >= gridDim.y/2) {
		//z = threadIdx.z + 2;
		y = threadIdx.y + (blockIdx.y - gridDim.y/2) * blockDim.y;
	}


	if( !(x > *DIM) &&  !(y > *DIM) && !(z > *DIM)) {//Guard against launching too many threads
	//set new cell state.
	
		//__syncthreads();

		g_data[(z * *DIM * *DIM) + (x * *DIM) + y] = func->applyFunction(g_data,x,y,z,*DIM);
		
	/*	g_data[0] = 1;*/
		//g_data[(x * *DIM) + y] = (x * *DIM) + y;
	}
}