#include "OuterTotalistic3dtest.h"

void OuterTotalistic3DTest::initTestCase() {
	
	xSize = 100;
	ySize = 100;
	zSize = 100;

	CACPU = new CellularAutomata_CPU();

	Lattice3D* latticeCPU = new Lattice3D(xSize,ySize,zSize,1);
	latticeCPU->setNeighbourhoodType(Lattice3D::MOORE_3D);
	
	genCPU = new OuterTotalistic3D();
	genCPU->setLattice(latticeCPU);

	genCPU->setStates(2);

	int* bornNumCPU = new int[1]; 
	genCPU->setBornNo(bornNumCPU,1);
	CACPU->setCARule(genCPU);

	CAGPU = new CellularAutomata_GPGPU();
	
	Lattice3D* latticeGPU = new Lattice3D(xSize,ySize,zSize,1);
	latticeGPU->setNeighbourhoodType(Lattice3D::MOORE_3D);
	
	genGPU = new OuterTotalistic3D();
	genGPU->setLattice(latticeGPU);

	genGPU->setStates(2);

	int* bornNumGPU = new int[1]; 
	genGPU->setBornNo(bornNumGPU,1);

	CAGPU->setCARule(genGPU);
}

void OuterTotalistic3DTest::cleanupTestCase(){
	delete CACPU;
	delete CAGPU;
}

void OuterTotalistic3DTest::performanceCPU(){
	cleanupTestCase();
	initTestCase();
	QBENCHMARK {
		 CACPU->nextTimeStep();
	}

}
void OuterTotalistic3DTest:: performanceGPU() {
	cleanupTestCase();
	initTestCase();
	QBENCHMARK {
		 CAGPU->nextTimeStep();
	 }
}

void OuterTotalistic3DTest::iterations(){
	cleanupTestCase();
	initTestCase();

	int noStates = genGPU->getNoStates();
	int noElements = genGPU->lattice->getNoElements();

	for(int i = 1; i < noStates; i++) {


		bool result = arrayEquals((unsigned int*)genGPU->lattice->pFlatGrid,noElements,i);
		QVERIFY(result);

		CAGPU->nextTimeStep();
	}

	//This timestep should set everything to 0.
	bool result = arrayEquals((unsigned int*)genGPU->lattice->pFlatGrid,noElements,0);
	QVERIFY(result);

}

bool OuterTotalistic3DTest::arrayEquals(unsigned int* grid, int size,int expectedVal){
	for(int i = 0; i < size; i++) {

		unsigned int val = grid[i];

		if(val != expectedVal) {
			return false;
		}
	}
	return true;
}