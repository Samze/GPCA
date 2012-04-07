#include "lattice2dtest.h"

void Lattice2DTest::initTestCase(){

	xDIM = 100;
	yDIM = 100;

	testLattice = new Lattice2D(xDIM,yDIM,1);


}

void Lattice2DTest::cleanupTestCase() {

	delete testLattice;

}

void Lattice2DTest::Lattice2DConstructorOne(){
	
	//Constructor one
	unsigned int xSize = 10;
	unsigned  int ySize = 20;

	int random = 1;

	Lattice2D* lattice = new Lattice2D(xSize,ySize,random);

	//Not null
	QVERIFY(lattice != NULL);

	//xSize
	QCOMPARE(lattice->getXSize(),xSize);
	//ySize
	 QCOMPARE(lattice->yDIM,ySize);
	
	//noElements
	 QCOMPARE(lattice->getNoElements(),xSize * ySize);

	//Test grid
	int* grid = (int*)lattice->getGrid();

	for(int i = 0; i < xSize * ySize; i++) {
		 QCOMPARE(grid[i],1);
	}

	delete lattice;

}

void Lattice2DTest::Lattice2DConstructorTwo(){
	
	//Constructor Two
	unsigned int xSize = 200;
	unsigned int ySize = 200;

	unsigned int* grid = new unsigned int[xSize * ySize];

	for(int i = 0; i < xSize*ySize; i++){
		grid[i] = i;
	}

	Lattice2D* lattice = new Lattice2D(grid,xSize,ySize);

	//Not null
	 QVERIFY(lattice != NULL);

	//xSize
	 QCOMPARE(lattice->getXSize(),xSize);

	//ySize
	 QCOMPARE(lattice->yDIM,ySize);
	
	//noElements
	 QCOMPARE(lattice->getNoElements(),xSize * ySize);
	
	//noElements
	 QCOMPARE((void*)grid,lattice->getGrid());

	delete lattice;
}


void Lattice2DTest::getMooresNeighbourhood() {
	int neighbourhoodStates[8];

	//set as -1 by default.
	for(int i = 0; i < 8; i++) {
		neighbourhoodStates[i] = -1; 
	}

	//0,0 
	testLattice->setNeighbourhoodType(Lattice2D::MOORE);

	//We should get back 3 neighs in moore.
	int expectedResult = 3;
	testLattice->getNeighbourhood(neighbourhoodStates,0,0,xDIM,yDIM);

	int countResults = 0;
	for(int i = 0; i < 8; i++) {

		if(neighbourhoodStates[i] != -1) {
			++countResults;
		}
	}
	 QCOMPARE(countResults,expectedResult);

	//set as -1 by default.
	for(int i = 0; i < 8; i++) {
		neighbourhoodStates[i] = -1; 
	}

	//We should get back 8 neighs in moore.
	expectedResult = 8;
	testLattice->getNeighbourhood(neighbourhoodStates,1,1,xDIM,yDIM);

	countResults = 0;
	for(int i = 0; i < 8; i++) {

		if(neighbourhoodStates[i] != -1) {
			++countResults;
		}
	}
	 QCOMPARE(countResults,expectedResult);

	//set as -1 by default.
	for(int i = 0; i < 8; i++) {
		neighbourhoodStates[i] = -1; 
	}

	//We should get back 5 neighs in moore.
	expectedResult = 5;
	testLattice->getNeighbourhood(neighbourhoodStates,0,2,xDIM,yDIM);

	countResults = 0;
	for(int i = 0; i < 8; i++) {

		if(neighbourhoodStates[i] != -1) {
			++countResults;
		}
	}
	 QCOMPARE(countResults,expectedResult);
	
	//set as -1 by default.
	for(int i = 0; i < 8; i++) {
		neighbourhoodStates[i] = -1; 
	}

	//We should get back 5 neighs in moore.
	expectedResult = 5;
	testLattice->getNeighbourhood(neighbourhoodStates,2,0,xDIM,yDIM);

	countResults = 0;
	for(int i = 0; i < 8; i++) {

		if(neighbourhoodStates[i] != -1) {
			++countResults;
		}
	}
	 QCOMPARE(countResults,expectedResult);

	
	//set as -1 by default.
	for(int i = 0; i < 8; i++) {
		neighbourhoodStates[i] = -1; 
	}

	//We should get back 8 neighs in moores
	//We are testing that the locations returned are correct.
	testLattice->getNeighbourhood(neighbourhoodStates,1,1,xDIM,yDIM);

	//One-One will lead to a location of
	int startingLoc = yDIM * 1 + 1;
	
	expectedResult = 8;
	countResults = 0;
	
	for(int i = 0; i < 8; i++) {

		int result = neighbourhoodStates[i];

		if(result == startingLoc - 1) { //-1,0
			++countResults;
		}
		else if(result == startingLoc + 1){ //1,0
			++countResults;
		}
		else if(result == startingLoc + yDIM + 1){ //1,1
			++countResults;
		}
		else if(result == startingLoc + yDIM - 1){ //-1,1
			++countResults;
		}
		else if(result == startingLoc - yDIM - 1){ //1,-1
			++countResults;
		}
		else if(result == startingLoc - yDIM + 1){ //1,-1
			++countResults;
		}
		else if(result == startingLoc - yDIM){ //0,-1
			++countResults;
		}
		else if(result == startingLoc + yDIM){ //0,1
			++countResults;
		}
		else {
			qDebug() << result;
		}
	}
	 QCOMPARE(countResults,expectedResult);
}


void Lattice2DTest::getVonNeumannNeighbourhood() {
	
	int neighbourhoodStates[4];

	//set as -1 by default.
	for(int i = 0; i < 4; i++) {
		neighbourhoodStates[i] = -1; 
	}

	//0,0 
	testLattice->setNeighbourhoodType(Lattice2D::VON_NEUMANN);

	//We should get back 2 neighs in von neu.
	int expectedResult = 2;
	testLattice->getNeighbourhood(neighbourhoodStates,0,0,xDIM,yDIM);

	int countResults = 0;
	for(int i = 0; i < 4; i++) {

		if(neighbourhoodStates[i] != -1) {
			++countResults;
		}
	}
	 QCOMPARE(countResults,expectedResult);

	//set as -1 by default.
	for(int i = 0; i < 4; i++) {
		neighbourhoodStates[i] = -1; 
	}

	//We should get back 4 neighs in von neu.
	expectedResult = 4;
	testLattice->getNeighbourhood(neighbourhoodStates,1,1,xDIM,yDIM);

	countResults = 0;
	for(int i = 0; i < 4; i++) {

		if(neighbourhoodStates[i] != -1) {
			++countResults;
		}
	}
	 QCOMPARE(countResults,expectedResult);

	//set as -1 by default.
	for(int i = 0; i < 4; i++) {
		neighbourhoodStates[i] = -1; 
	}

	//We should get back 3 neighs in von neu.
	expectedResult = 3;
	testLattice->getNeighbourhood(neighbourhoodStates,0,2,xDIM,yDIM);

	countResults = 0;
	for(int i = 0; i < 4; i++) {

		if(neighbourhoodStates[i] != -1) {
			++countResults;
		}
	}
	 QCOMPARE(countResults,expectedResult);
	
	//set as -1 by default.
	for(int i = 0; i < 4; i++) {
		neighbourhoodStates[i] = -1; 
	}

	//We should get back 3 neighs in von neu.
	expectedResult = 3;
	testLattice->getNeighbourhood(neighbourhoodStates,2,0,xDIM,yDIM);

	countResults = 0;
	for(int i = 0; i < 4; i++) {

		if(neighbourhoodStates[i] != -1) {
			++countResults;
		}
	}
	 QCOMPARE(countResults,expectedResult);

	
	//set as -1 by default.
	for(int i = 0; i < 4; i++) {
		neighbourhoodStates[i] = -1; 
	}


	//We are testing that the locations returned are correct.
	testLattice->getNeighbourhood(neighbourhoodStates,1,1,xDIM,yDIM);

	//One-One will lead to a location of
	int startingLoc = yDIM * 1 + 1;
	
	expectedResult = 4;
	countResults = 0;
	
	for(int i = 0; i < 4; i++) {

		int result = neighbourhoodStates[i];

		if(result == startingLoc - 1) { //-1,0
			++countResults;
		}
		else if(result == startingLoc + 1){ //1,0
			++countResults;
		}
		else if(result == startingLoc - yDIM){ //0,-1
			++countResults;
		}
		else if(result == startingLoc + yDIM){ //0,1
			++countResults;
		}
		else {
			qDebug() << result;
		}
	}
	 QCOMPARE(countResults,expectedResult);
}