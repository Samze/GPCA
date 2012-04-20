#include "SCIARAPerformanceTest.h"

void SCIARAPerformanceTest::testSCIARACPU() {

	int size = 1000;
	SCIARA::Cell* cGrid = new SCIARA::Cell[size * size];

	for(int row = 0; row < size; row++) {
		for(int col = 0; col < size; col++) {
			int address = col * size + row;

			int rnd = std::rand () % 10; // add random lava

			cGrid[address].altitude = 100;	
			cGrid[address].thickness = rnd;
			cGrid[address].outflow[0] = 0;
			cGrid[address].outflow[1] = 0;
			cGrid[address].outflow[2] = 0;
			cGrid[address].outflow[3] = 0;
		}
	}

	//Create CA
	Lattice2D* ab2D = new Lattice2D(size,size,10);
	ab2D->pFlatGrid = cGrid;

	ab2D->setNeighbourhoodType(ab2D->VON_NEUMANN);

	SCIARA* sciara = new SCIARA();
	sciara->setLattice(ab2D);

	QBENCHMARK {
		testType(sciara, CPU);

		SCIARAThickness* sciaraThickness = new SCIARAThickness();

		sciaraThickness->setLattice(ab2D);

		testType(sciaraThickness,CPU);
	}
}

void SCIARAPerformanceTest::testSCIARAGPU() {
	int size = 1000;
	SCIARA::Cell* cGrid = new SCIARA::Cell[size * size];

	for(int row = 0; row < size; row++) {
		for(int col = 0; col < size; col++) {
			int address = col * size + row;

			int rnd = std::rand () % 10; // add random lava

			cGrid[address].altitude = 100;	
			cGrid[address].thickness = rnd;
			cGrid[address].outflow[0] = 0;
			cGrid[address].outflow[1] = 0;
			cGrid[address].outflow[2] = 0;
			cGrid[address].outflow[3] = 0;
		}
	}

	//Create CA
	Lattice2D* ab2D = new Lattice2D(size,size,10);
	ab2D->pFlatGrid = cGrid;

	ab2D->setNeighbourhoodType(ab2D->VON_NEUMANN);

	SCIARA* sciara = new SCIARA();
	sciara->setLattice(ab2D);
	
	QBENCHMARK {
		testType(sciara, GPU);

		SCIARAThickness* sciaraThickness = new SCIARAThickness();

		sciaraThickness->setLattice(ab2D);

		testType(sciaraThickness,GPU);

	}
}



void SCIARAPerformanceTest::testType(AbstractCellularAutomata* rule, Runtype runtype) {
	
	CellularAutomata* run;
	
	if(runtype == Runtype::CPU) {
		run = new CellularAutomata_CPU();
	}
	else {
		run = new CellularAutomata_GPGPU();
	}

	run->setCARule(rule);

	run->nextTimeStep();

	//DELETE run, this will break the test
}