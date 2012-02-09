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
	
	int x = threadIdx.x + blockIdx.x * blockDim.x; 
    int y = threadIdx.y + blockIdx.y * blockDim.y;
	
	int DIM = func->lattice->DIM;
	unsigned int* grid = func->lattice->pFlatGrid;

	if( !(x > DIM) &&  !(y > DIM)) {//Guard against launching too many threads
	//set new cell state.
	
		//__syncthreads();

		grid[(x * DIM) + y] = func->applyFunction(grid,x,y,DIM);
	
		//g_data[(x * *DIM) + y] = (x * *DIM) + y;
	}
}

template <typename CAFunction>
__global__ void kernal3DTest(CAFunction* func) {
	int DIM = func->lattice->DIM;
	unsigned int* grid = func->lattice->pFlatGrid;

	int x = threadIdx.x + blockIdx.x * blockDim.x; 
	
	int slice = DIM/blockDim.y + 1;

	int y = (blockIdx.y % slice) * blockDim.y + threadIdx.y;
	int z = blockIdx.y/slice;


	if( x >= DIM ||  y >= DIM || z >= DIM) //Guard against launching too many threads
		return;
	
	//__syncthreads();

	grid[(z * DIM * DIM) + (x * DIM) + y] = func->applyFunction(grid,x,y,z,DIM);
	
}

//template <typename CAFunction>
//__global__ void kernal3D(unsigned int* g_data, int* DIM, CAFunction* func) {
//	
//	int x = threadIdx.x + blockIdx.x * blockDim.x; 
//    int y = threadIdx.y + blockIdx.y * blockDim.y;
//	int z = threadIdx.z;// + blockIdx.z * blockDim.z;
//
//	//This is our fake z area
//	if( blockIdx.x >= gridDim.x/2) {
//		//z = threadIdx.z + blockDim.z;
//		x = threadIdx.x + (blockIdx.x - gridDim.x/2) * blockDim.x;
//	}
//
//	if( blockIdx.y >= gridDim.y/2) {
//		//z = threadIdx.z + 2;
//		y = threadIdx.y + (blockIdx.y - gridDim.y/2) * blockDim.y;
//	}
//
//
//	if( !(x > *DIM) &&  !(y > *DIM) && !(z > *DIM)) {//Guard against launching too many threads
//	//set new cell state.
//	
//		//__syncthreads();
//
//		g_data[(z * *DIM * *DIM) + (x * *DIM) + y] = func->applyFunction(g_data,x,y,z,*DIM);
//		
//	/*	g_data[0] = 1;*/
//		//g_data[(x * *DIM) + y] = (x * *DIM) + y;
//	}
//}
