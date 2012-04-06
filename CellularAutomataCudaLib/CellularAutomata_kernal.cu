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
template <typename CAFunction>
__global__ void kernal(CAFunction* func) {
	
	//global x/y positions
	int x = threadIdx.x + blockIdx.x * blockDim.x; 
    int y = threadIdx.y + blockIdx.y * blockDim.y;


	int xDIM = func->lattice->getXSize();
	int yDIM = func->lattice->yDIM;

	void* grid = func->lattice->getGrid();
	

	if( !(x > xDIM) &&  !(y > yDIM)) {//Guard against launching too many threads
			
		func->applyFunction(grid,x,y,xDIM,yDIM);
		//grid[x * DIM + y] = result;
	}
}

template <typename CAFunction>
__global__ void kernalSharedMem(CAFunction* func) {
	
	//global x/y positions
	int x = threadIdx.x + blockIdx.x * blockDim.x; 
    int y = threadIdx.y + blockIdx.y * blockDim.y;

	//block x/y positions (adjusting for launching more threads than needed for the update of states)
	int bx = threadIdx.x - 1;
	int by = threadIdx.y - 1;

	//int xDIM = func->lattice->xDIM;
	int yDIM = func->lattice->yDIM;

	//22 needs to mirror how many threads launched in the kernal...can't use blockDim.x/y.
	__shared__ unsigned int shar_data[22 * 22];

	unsigned int* grid = (unsigned int*)func->lattice->getGrid();
	
	//2 because of the padding!
	int xOrigin = (x - 1) - (blockIdx.x * 2);
	int yOrigin = (y - 1) - (blockIdx.y * 2);
	
	int bounds = (blockDim.x * gridDim.y) - 1;
	//checking bounds..
	if(x != 0 && y != 0 && x != bounds && y != bounds) {

		shar_data[threadIdx.x * blockDim.x + threadIdx.y] = grid[xOrigin * yDIM + yOrigin];

	} else {
		shar_data[threadIdx.x * blockDim.x + threadIdx.y] = 0;
	}

	//void* grid = func->lattice->pFlatGrid;
	__syncthreads();

	
	//We only want to update the state of cells in our 'inner area'
	if(bx >= 0 && bx < 20 && by >= 0 && by < 20) {

		func->applyFunction(shar_data,threadIdx.x,threadIdx.y,blockDim.x,blockDim.y);
		
		//__syncthreads();
		grid[xOrigin * yDIM + yOrigin] = shar_data[threadIdx.x * blockDim.x + threadIdx.y];

	}
}


template <typename CAFunction>
__global__ void SCIARAKernal(CAFunction* func) {
	
	int x = threadIdx.x + blockIdx.x * blockDim.x; 
    int y = threadIdx.y + blockIdx.y * blockDim.y;
	
	int xDIM = func->lattice->getXSize();	
	int yDIM = func->lattice->yDIM;

	void* grid =  func->lattice->getGrid();

	if( !(x > xDIM) &&  !(y > yDIM)) {//Guard against launching too many threads
	//set new cell state.
	
		//__syncthreads();

		func->applyFunction(grid,x,y,xDIM,yDIM);
	
		//__syncthreads();
		
		//func->computethickness(grid,x,y,xDIM,yDIM);

		//grid[(x * DIM) + y] = (976562499 << func->lattice->noBits);
		//g_data[(x * *DIM) + y] = (x * *DIM) + y;
	}
}


template <typename CAFunction>
__global__ void kernal3D(CAFunction* func) {
	int xDIM = func->lattice->getXSize();	
	int yDIM = func->lattice->yDIM;
	int zDIM = func->lattice->zDIM;

	void* grid = (unsigned int*)func->lattice->getGrid();

	int x = threadIdx.x + blockIdx.x * blockDim.x; 
	
	int slice = xDIM/blockDim.y + 1;

	int y = (blockIdx.y % slice) * blockDim.y + threadIdx.y;
	int z = blockIdx.y/slice;

	//TODO fix coding style inconsistancy.
	if( x >= xDIM ||  y >= yDIM || z >= zDIM) //Guard against launching too many threads
		return;
	

//	grid[(z * DIM * DIM) + (x * DIM) + y] = func->applyFunction(grid,x,y,z,DIM);
	func->applyFunction(grid,x,y,z,xDIM);
	
}

template <typename CAFunction>
__global__ void kernal3DTest(CAFunction* func) {
	int xDIM = func->lattice->getXSize();	
	int yDIM = func->lattice->yDIM;
	int zDIM = func->lattice->zDIM;

	unsigned int* grid = (unsigned int*)func->lattice->getGrid();

	int blockSlice = blockIdx.x / gridDim.y;
	
	__shared__ unsigned int shar_data[512];

	int x = threadIdx.x + (blockIdx.x - (blockSlice * gridDim.y))  * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y; 
	int z = threadIdx.z + blockSlice * blockDim.x;



	//TODO fix coding style inconsistancy.
	if( x >= xDIM ||  y >= yDIM || z >= zDIM) //Guard against launching too many threads
		return;

	func->applyFunction(grid,x,y,z,xDIM);
}


template <typename CAFunction>
__global__ void kernal3DTestShared(CAFunction* func) {
	int xDIM = func->lattice->getXSize();	
	int yDIM = func->lattice->yDIM;
	int zDIM = func->lattice->zDIM;

	unsigned int* grid = (unsigned int*)func->lattice->getGrid();

	int blockSlice = blockIdx.x / gridDim.y;

	int x = threadIdx.x + (blockIdx.x - (blockSlice * gridDim.y))  * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y; 
	int z = threadIdx.z + blockSlice * blockDim.z;

	//8*8*8, 512 is the maximum size of a block..
	__shared__ unsigned int shar_data[512];

	//2 because of the padding!
	int xOrigin = (x - 1) - ((blockIdx.x - (blockSlice * gridDim.y)) * 2);
	int yOrigin = (y - 1) - (blockIdx.y * 2);
	int zOrigin = (z - 1) - (blockSlice * 2);


	int sharPos = (threadIdx.z * blockDim.z * blockDim.z) + (threadIdx.x * blockDim.x) + threadIdx.y;	

	int globPos = (zOrigin * zDIM * zDIM) + (xOrigin * xDIM) + yOrigin;

	int bounds = (blockDim.x * gridDim.y) - 1;
	
	////checking bounds..
	if(x != 0 && y != 0 && z != 0 && x !=  bounds && y != bounds  && z != bounds) {
		shar_data[sharPos] = grid[globPos];
	} 
	else{
		shar_data[sharPos] = 0;
	}
	
	__syncthreads();

	////We only want to update the state of cells in our 'inner area'
	if((threadIdx.x - 1) >= 0 && (threadIdx.x - 1) < 6 && (threadIdx.y - 1) >= 0 && (threadIdx.y - 1) < 6 && (threadIdx.z - 1) >= 0 && (threadIdx.z - 1) < 6) {
		
		func->applyFunction(shar_data,threadIdx.x,threadIdx.y,threadIdx.z,blockDim.y);
		grid[globPos] = shar_data[sharPos];

	}
}