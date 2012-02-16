#pragma once

#include <device_launch_parameters.h>
#include <device_functions.h>
#include "abstractcellularautomata.h"
#include "Abstract2DCA.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include <map>
#include <vector>

using namespace std;

class SCIARA :
	public AbstractCellularAutomata
{
public:
	DLLExport __device__ __host__ SCIARA(void);
	DLLExport __device__ __host__  ~SCIARA(void);

	Abstract2DCA *lattice;

	//For now this must be signed to cope with -1 (no neighbour values)
	__device__ struct Cell {
	  int altitude;
	  int thickness;
	  int outflow[4];
	};

	__host__ __device__ virtual AbstractLattice* getLattice() { return lattice;}


	__host__ map<void**, size_t>* getDynamicArrays() {
		
		map<void**, size_t>* newMap = new map<void**, size_t>();

		size_t gridMemSize = lattice->DIM * lattice->DIM * lattice->DIM * sizeof(unsigned int);

		newMap->insert(make_pair((void**)&lattice->pFlatGrid, gridMemSize));

		return newMap;
	}

	/* State information.

	z = 0-1000;
	h = 0-100;
	f[4] = 0-100 (400)

	max = 1000 * 100 * 800;
	*/
	__device__  int applyFunction(unsigned int* g_data, int x, int y, int DIM) { 

		int xAltered = x * DIM;
		int gridLoc = x * DIM + y;

		//cuda sm1.1 does not support recursion, shame.
		int originalState = g_data[gridLoc]; 
		unsigned int state = originalState;
		unsigned int total = 100 * 25 * 25 * 25 * 25 * 25;

		int altitude = state / (total/100);
		
		total = total/100;
		state = state - (altitude * total);

		int thickness = state / (total/25);

		total = total/25;
		state = state - (thickness * total);

		int flowN = state / (total/25);

		total = total/25;
		state = state - (flowN * total);

		int flowE = state / (total/25);

		total = total/25;
		state = state - (flowE * total);

		int flowS = state / (total/25);

		total = total/25;
		state = state - (flowS * total);

		int flowW = state / (total/25);

		Cell center;
		center.altitude = altitude;
		center.thickness = thickness;
		center.outflow[0] = flowN;
		center.outflow[1] = flowE;
		center.outflow[2] = flowS;
		center.outflow[3] = flowW;
		
		Cell neighs[4];

		int neighbourhoodStates[4];
	//
	//	//set as -1 by default.
		for(int i = 0; i < 8; i++) {
			neighbourhoodStates[i] = -1; 
		}

		lattice->getNeighbourhood(neighbourhoodStates,g_data,xAltered,y);

		//Populate neighbours
		for(int i = 0; i < 4; i++) {
			int state = neighbourhoodStates[i];
			
			if (state != -1) {
				unsigned int total = 100 * 25 * 25 * 25 * 25 * 25;

				int neighAltitude = state / (total/100);
		
				total = total/100;
				state = state - (neighAltitude * total);

				int neighThickness = state / (total/25);

				total = total/25;
				state = state - (neighThickness * total);

				int neighFlowN = state / (total/25);

				total = total/25;
				state = state - (neighFlowN * total);

				int neighFlowE = state / (total/25);

				total = total/25;
				state = state - (neighFlowE * total);

				int neighFlowS = state / (total/25);

				total = total/25;
				state = state - (neighFlowS * total);

				int neighFlowW = state / (total/25);

				neighs[i].altitude = neighAltitude;
				neighs[i].thickness = neighThickness;
				neighs[i].outflow[0] = neighFlowN;
				neighs[i].outflow[1] = neighFlowE;
				neighs[i].outflow[2] = neighFlowS;
				neighs[i].outflow[3] = neighFlowW;
			}
			else {
				neighs[i].altitude = 100; //set to max height
				neighs[i].thickness = 0;
				neighs[i].outflow[0] = 0;
				neighs[i].outflow[1] = 0;
				neighs[i].outflow[2] = 0;
				neighs[i].outflow[3] = 0;
			}
		}

		bool elim[5] = {false,false,false,false,false};
		//Mobile part of center
		float m;

		float Z[5];

		float average;

		int k,i;
		bool again;
	
		m = thickness;

		Z[0] = altitude;

		for(int i = 1; i < 5; i++) {
			Z[i] = neighs[i-1].altitude + neighs[i-1].thickness;
		}


		do {
			again = false;
			k = 0;
			average = m;

			for(i = 0;  i<5 ; i++) {
				if ( elim[i] == false) {
					average += Z[i];
					k++;
				}
			}
			average = average/k;

			for(i=0;i<5;i++) {
				if ((average <= Z[i]) && elim[i] == false) {
					elim[i] = true;
					again = true;
				}
			}
		}
		while(again);

		for (i = 1; i < 5; i++) {
			if (elim[i] == false) {
				center.outflow[i-1] = (average - Z[i]) * 0.7;
			}
			else {
				center.outflow[i-1] = 0;
			}
		}

		total = 100 * 25 * 25 * 25 * 25 * 25;

		unsigned int newState2 =  (center.altitude * (total/100)) + (center.thickness * (total/ (100 * 25)))  + (center.outflow[0]  * (total/ (100 * 25 * 25))) + (center.outflow[1] * (total/ (100 * 25 * 25 * 25))) + (center.outflow[2] * (total/ (100 * 25 * 25 * 25 * 25))) + (center.outflow[3]  * (total/ (100 * 25 * 25 * 25 * 25 * 25)));
		return newState2;//setNewState(lattice,newState,originalState);

	}

	__device__  int computethickness(unsigned int* g_data, int x, int y, int DIM) { 
		
		int xAltered = x * DIM;
		int gridLoc = x * DIM + y;

		//cuda sm1.1 does not support recursion, shame.
		int originalState = g_data[gridLoc];
		int state = originalState;
		int total = 100 * 25 * 25 * 25 * 25 * 25;

		int altitude = state / (total/100);
		
		total = total/100;
		state = state - (altitude * total);

		int thickness = state / (total/25);

		total = total/25;
		state = state - (thickness * total);

		int flowN = state / (total/25);

		total = total/25;
		state = state - (flowN * total);

		int flowE = state / (total/25);

		total = total/25;
		state = state - (flowE * total);

		int flowS = state / (total/25);

		total = total/25;
		state = state - (flowS * total);

		int flowW = state / (total/25);

		Cell center;
		center.altitude = altitude;
		center.thickness = thickness;
		center.outflow[0] = flowN;
		center.outflow[1] = flowE;
		center.outflow[2] = flowS;
		center.outflow[3] = flowW;


		int neighbourhoodStates[4];
	//
	//	//set as -1 by default.
		for(int i = 0; i < 8; i++) {
			neighbourhoodStates[i] = -1; 
		}

		lattice->getNeighbourhood(neighbourhoodStates,g_data,xAltered,y);
		
		Cell neighs[4];
		//Populate neighbours

		for(int i = 0; i < 4; i++) {
			unsigned int state = neighbourhoodStates[i];

			if (state != -1) {

				unsigned int total = 100 * 25 * 25 * 25 * 25 * 25;

				int neighAltitude = state / (total/100);
		
				total = total/100;
				state = state - (neighAltitude * total);

				int neighThickness = state / (total/25);

				total = total/25;
				state = state - (neighThickness * total);

				int neighFlowN = state / (total/25);

				total = total/25;
				state = state - (neighFlowN * total);

				int neighFlowE = state / (total/25);

				total = total/25;
				state = state - (neighFlowE * total);

				int neighFlowS = state / (total/25);

				total = total/25;
				state = state - (neighFlowS * total);

				int neighFlowW = state / (total/25);

				neighs[i].altitude = neighAltitude;
				neighs[i].thickness = neighThickness;
				neighs[i].outflow[0] = neighFlowN;
				neighs[i].outflow[1] = neighFlowE;
				neighs[i].outflow[2] = neighFlowS;
				neighs[i].outflow[3] = neighFlowW;
			}
			else {
				neighs[i].altitude = 100; //set to max height
				neighs[i].thickness = 0;
				neighs[i].outflow[0] = 0;
				neighs[i].outflow[1] = 0;
				neighs[i].outflow[2] = 0;
				neighs[i].outflow[3] = 0;
			}

		}

		int new_thickness;
		int i;

		int outflows = 0;

		new_thickness = center.thickness;

		for(i=0; i < 4; i++) {
			new_thickness = new_thickness + center.outflow[i] + neighs[i].outflow[3-i];
			outflows += center.outflow[i];
		}
		center.thickness = new_thickness;

		total = 100 * 25 * 25 * 25 * 25 * 25;

		unsigned int newState2 =  (center.altitude * (total/100)) + (center.thickness * (total/ (100 * 25)))  + (center.outflow[0]  * (total/ (100 * 25 * 25))) + (center.outflow[1] * (total/ (100 * 25 * 25 * 25))) + (center.outflow[2] * (total/ (100 * 25 * 25 * 25 * 25))) + (center.outflow[3]  * (total/ (100 * 25 * 25 * 25 * 25 * 25)));
		return newState2;//setNewState(lattice,newState,originalState);
	}

};

