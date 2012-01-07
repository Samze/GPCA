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