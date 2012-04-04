#ifndef CELLULAR_AUTOMATATA_LAUNCHER
#define CELLULAR_AUTOMATATA_LAUNCHER

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vector_types.h>
#include <map>
#include <vector>

using namespace std;

void test();
vector<void*>* setupDynamicArrays(const map<void**, size_t> &myMap);
const char* errorCheck();

#endif