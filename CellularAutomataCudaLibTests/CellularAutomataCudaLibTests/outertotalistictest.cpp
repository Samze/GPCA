#include "outertotalistictest.h"

void OuterTotalisticTest::initTestCase() {
	
	xSize = 500;
	ySize = 500;

	CACPU = new CellularAutomata_CPU();

	Lattice2D* latticeCPU = new Lattice2D(xSize,ySize,1);
	latticeCPU->setNeighbourhoodType(Lattice2D::MOORE);
	
	genCPU = new OuterTotalistic();
	genCPU->setLattice(latticeCPU);

	genCPU->setStates(2);

	int* bornNumCPU = new int[1]; 
	genCPU->setBornNo(bornNumCPU,1);
	CACPU->setCARule(genCPU);

	CAGPU = new CellularAutomata_GPGPU();
	
	Lattice2D* latticeGPU = new Lattice2D(xSize,ySize,1);
	latticeGPU->setNeighbourhoodType(Lattice2D::MOORE);
	
	genGPU = new OuterTotalistic();
	genGPU->setLattice(latticeGPU);

	genGPU->setStates(2);

	int* bornNumGPU = new int[1]; 
	genGPU->setBornNo(bornNumGPU,1);

	CAGPU->setCARule(genGPU);
}

void OuterTotalisticTest::cleanupTestCase(){
	delete CACPU;
	delete CAGPU;
}

void OuterTotalisticTest::performanceCPU(){
	cleanupTestCase();
	initTestCase();
	QBENCHMARK {
		 CACPU->nextTimeStep();
	}

}
void OuterTotalisticTest:: performanceGPU() {
	cleanupTestCase();
	initTestCase();
	QBENCHMARK {
		 CAGPU->nextTimeStep();
	 }
}

void OuterTotalisticTest::iterations(){
	cleanupTestCase();
	initTestCase();

	int noStates = genGPU->getNoStates();

	for(int i = 1; i < noStates; i++) {
		
		bool result = arrayEquals((unsigned int*)genGPU->lattice->pFlatGrid, xSize * ySize,i);
		QVERIFY(result);

		CAGPU->nextTimeStep();
	}

	//This timestep should set everything to 0.
	bool result = arrayEquals((unsigned int*)genGPU->lattice->pFlatGrid, xSize * ySize,0);
	QVERIFY(result);

}

bool OuterTotalisticTest::arrayEquals(unsigned int* grid, int size,int expectedVal){
	for(int i = 0; i < size; i++) {

		unsigned int val = grid[i];

		if(val != expectedVal) {
			return false;
		}
	}
	return true;
}