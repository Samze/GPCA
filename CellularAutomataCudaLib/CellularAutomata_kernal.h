#ifndef CELLULARAUTOMATA_KERNAL_DLL_H
#define CELLULARAUTOMATA_KERNAL_DLL_H

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#include "gameoflifetransition.h"

template <typename CAFunction> __global__ void kernal(int* g_data, int* DIM, CAFunction* func);

#endif