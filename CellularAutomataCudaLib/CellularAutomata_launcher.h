#ifndef CELLULAR_AUTOMATATA_LAUNCHER
#define CELLULAR_AUTOMATATA_LAUNCHER

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vector_types.h>
#include <map>
#include <vector>

using namespace std;

/**
* Utility class that provides the ability to dynimcally allocate memory and copy data to the GPU. This function is used internally
* and should not be used by any clients.
*/
vector<void*>* setupDynamicArrays(const map<void**, size_t> &myMap);

/**
* Provides error checking on any potential GPU issues.
*/
const char* errorCheck();

#endif