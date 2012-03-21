
#include "CellularAutomata_launcher.h"


void test() {
}

vector<void*>* setupDynamicArrays(const map<void**, size_t> &myMap) {

	map<void**, size_t>::const_iterator iter;

	vector<void*>* devPointers = new vector<void*>();

	for(iter = myMap.begin(); iter != myMap.end(); ++iter) {
		
		void* host_pointer = *(*iter).first;
		size_t size = (*iter).second;

		void* dev_pointer;
		
		//alloc memory
		cudaMalloc((void**) &dev_pointer, size);
	
		//copy data
		cudaMemcpy(dev_pointer, host_pointer, size,
			cudaMemcpyHostToDevice);

		devPointers->push_back(dev_pointer);
	}

	errorCheck();
	return devPointers;
}

//TODO add support for this.
const char* errorCheck() {
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		return cudaGetErrorString(err);
	}
	return  NULL;
}
