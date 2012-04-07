#include "lattice3dtest.h"

void Lattice3DTest::initTestCase(){

	xDIM = 100;
	yDIM = 100;
	zDIM = 100;

	testLattice = new Lattice3D(xDIM,yDIM,zDIM,1);
   
}

void Lattice3DTest::cleanupTestCase() {

	delete testLattice;

}

void Lattice3DTest::Lattice3DConstructorOne(){
	
	//Constructor one
	unsigned int xSize = 20;
	unsigned  int ySize = 20;
	unsigned  int zSize = 1;

	int random = 1;

	Lattice3D* lattice = new Lattice3D(xSize,ySize,zSize,random);

	//Not null
	QVERIFY(lattice != NULL);

	//xSize
	QCOMPARE(lattice->getXSize(),xSize);

	//ySize
	 QCOMPARE(lattice->yDIM,ySize);
	
	//zSize
	 QCOMPARE(lattice->zDIM,zSize);
	
	//noElements
	 QCOMPARE(lattice->getNoElements(),xSize * ySize * zSize);

	//Test grid
	int* grid = (int*)lattice->getGrid();

	for(int i = 0; i < xSize * ySize * zSize; i++) {
		 QCOMPARE(grid[i],1);
	}

	delete lattice;

}

void Lattice3DTest::Lattice3DConstructorTwo(){
	
	//Constructor Two
	unsigned int xSize = 200;
	unsigned int ySize = 200;
	unsigned int zSize = 200;

	unsigned int* grid = new unsigned int[xSize * ySize * zSize];

	for(int i = 0; i < xSize*ySize; i++){
		grid[i] = i;
	}

	Lattice3D* lattice = new Lattice3D(grid,xSize,ySize,zSize);

	//Not null
	 QVERIFY(lattice != NULL);

	//xSize
	 QCOMPARE(lattice->getXSize(),xSize);

	//ySize
	 QCOMPARE(lattice->yDIM,ySize);
	 
	//zSize
	 QCOMPARE(lattice->zDIM,zSize);
	
	//noElements
	 QCOMPARE(lattice->getNoElements(),xSize * ySize * zSize);
	
	//noElements
	 QCOMPARE((void*)grid,lattice->getGrid());

	delete lattice;
}


void Lattice3DTest::getMooresNeighbourhood() {
	int neighbourhoodStates[26];

	//set as -1 by default.
	for(int i = 0; i < 26; i++) {
		neighbourhoodStates[i] = -1; 
	}

	//0,0,0
	testLattice->setNeighbourhoodType(Lattice3D::MOORE_3D);

	//We should get back 3 neighs in moore.
	int expectedResult = 7;
	testLattice->getNeighbourhood(neighbourhoodStates,0,0,0,xDIM);

	int countResults = 0;
	for(int i = 0; i < 26; i++) {

		if(neighbourhoodStates[i] != -1) {
			++countResults;
		}
	}
	 QCOMPARE(countResults,expectedResult);

	//set as -1 by default.
	for(int i = 0; i < 26; i++) {
		neighbourhoodStates[i] = -1; 
	}

	//We should get back 26 neighs in moore.
	expectedResult = 26;
	testLattice->getNeighbourhood(neighbourhoodStates,1,1,1,xDIM);

	countResults = 0;
	for(int i = 0; i < 26; i++) {

		if(neighbourhoodStates[i] != -1) {
			++countResults;
		}
	}
	 QCOMPARE(countResults,expectedResult);

	//set as -1 by default.
	for(int i = 0; i < 26; i++) {
		neighbourhoodStates[i] = -1; 
	}

	//We should get back 5 neighs in moore.
	expectedResult = 11;
	testLattice->getNeighbourhood(neighbourhoodStates,0,2,0,xDIM);

	countResults = 0;
	for(int i = 0; i < 26; i++) {

		if(neighbourhoodStates[i] != -1) {
			++countResults;
		}
	}
	 QCOMPARE(countResults,expectedResult);
	
	//set as -1 by default.
	for(int i = 0; i < 26; i++) {
		neighbourhoodStates[i] = -1; 
	}

	//We should get back 5 neighs in moore.
	expectedResult = 11;
	testLattice->getNeighbourhood(neighbourhoodStates,2,0,0,xDIM);

	countResults = 0;
	for(int i = 0; i < 26; i++) {

		if(neighbourhoodStates[i] != -1) {
			++countResults;
		}
	}
	 QCOMPARE(countResults,expectedResult);

	
	//set as -1 by default.
	for(int i = 0; i < 26; i++) {
		neighbourhoodStates[i] = -1; 
	}

	//We should get back 26 neighs in moores
	//We are testing that the locations returned are correct.
	testLattice->getNeighbourhood(neighbourhoodStates,1,1,1,xDIM);

	//One-One will lead to a location of
	int sqzDIM = zDIM * zDIM;
	int startingLoc = sqzDIM + yDIM + 1;
	
	expectedResult = 26;
	countResults = 0;
	
	for(int i = 0; i < 26; i++) {
		
		//z = 0;
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
		
		//z = -1;
		else if(result == startingLoc - 1 - sqzDIM) { //-1,0
			++countResults;
		}
		else if(result == startingLoc + 1 - sqzDIM){ //1,0
			++countResults;
		}
		else if(result == startingLoc + yDIM + 1 - sqzDIM){ //1,1
			++countResults;
		}
		else if(result == startingLoc + yDIM - 1 - sqzDIM){ //-1,1
			++countResults;
		}
		else if(result == startingLoc - yDIM - 1 - sqzDIM){ //1,-1
			++countResults;
		}
		else if(result == startingLoc - yDIM + 1 - sqzDIM){ //1,-1
			++countResults;
		}
		else if(result == startingLoc - yDIM - sqzDIM){ //0,-1
			++countResults;
		}
		else if(result == startingLoc + yDIM - sqzDIM){ //0,1
			++countResults;
		}

		else if(result == startingLoc - sqzDIM){ //0,1
			++countResults;
		}


		//z =+1
		else if(result == startingLoc - 1 + sqzDIM) { //-1,0
			++countResults;
		}
		else if(result == startingLoc + 1 + sqzDIM){ //1,0
			++countResults;
		}
		else if(result == startingLoc + yDIM + 1 + sqzDIM){ //1,1
			++countResults;
		}
		else if(result == startingLoc + yDIM - 1 + sqzDIM){ //-1,1
			++countResults;
		}
		else if(result == startingLoc - yDIM - 1 + sqzDIM){ //1,-1
			++countResults;
		}
		else if(result == startingLoc - yDIM + 1 + sqzDIM){ //1,-1
			++countResults;
		}
		else if(result == startingLoc - yDIM + sqzDIM){ //0,-1
			++countResults;
		}
		else if(result == startingLoc + yDIM + sqzDIM){ //0,1
			++countResults;
		}

		else if(result == startingLoc + sqzDIM){ //0,1
			++countResults;
		}

		else {
			qDebug() << result;
		}
	}
	 QCOMPARE(countResults,expectedResult);
}


void Lattice3DTest::getVonNeumannNeighbourhood() {
	
	int neighbourhoodStates[6];

	//set as -1 by default.
	for(int i = 0; i < 6; i++) {
		neighbourhoodStates[i] = -1; 
	}

	//0,0 
	testLattice->setNeighbourhoodType(Lattice3D::VON_NEUMANN_3D);

	//We should get back 3 neighs in von neu.
	int expectedResult = 3;
	testLattice->getNeighbourhood(neighbourhoodStates,0,0,0,xDIM);

	int countResults = 0;
	for(int i = 0; i < 6; i++) {

		if(neighbourhoodStates[i] != -1) {
			++countResults;
		}
	}
	 QCOMPARE(countResults,expectedResult);

	//set as -1 by default.
	for(int i = 0; i < 6; i++) {
		neighbourhoodStates[i] = -1; 
	}

	//We should get back 6 neighs in von neu.
	expectedResult = 6;
	testLattice->getNeighbourhood(neighbourhoodStates,1,1,1,xDIM);

	countResults = 0;
	for(int i = 0; i < 6; i++) {

		if(neighbourhoodStates[i] != -1) {
			++countResults;
		}
	}
	 QCOMPARE(countResults,expectedResult);

	//set as -1 by default.
	for(int i = 0; i < 6; i++) {
		neighbourhoodStates[i] = -1; 
	}

	//We should get back 4 neighs in von neu.
	expectedResult = 4;
	testLattice->getNeighbourhood(neighbourhoodStates,0,2,0,xDIM);

	countResults = 0;
	for(int i = 0; i < 6; i++) {

		if(neighbourhoodStates[i] != -1) {
			++countResults;
		}
	}
	 QCOMPARE(countResults,expectedResult);
	
	//set as -1 by default.
	for(int i = 0; i < 6; i++) {
		neighbourhoodStates[i] = -1; 
	}

	//We should get back 4 neighs in von neu.
	expectedResult = 4;
	testLattice->getNeighbourhood(neighbourhoodStates,2,0,0,xDIM);

	countResults = 0;
	for(int i = 0; i < 6; i++) {

		if(neighbourhoodStates[i] != -1) {
			++countResults;
		}
	}
	 QCOMPARE(countResults,expectedResult);

	
	//set as -1 by default.
	for(int i = 0; i < 6; i++) {
		neighbourhoodStates[i] = -1; 
	}


	//We should get back 26 neighs in moores
	//We are testing that the locations returned are correct.
	testLattice->getNeighbourhood(neighbourhoodStates,1,1,1,xDIM);

	//One-One will lead to a location of
	int sqzDIM = zDIM * zDIM;
	int startingLoc = sqzDIM + yDIM + 1;
	
	expectedResult = 6;
	countResults = 0;
	
	for(int i = 0; i < 6; i++) {
		
		//z = 0;
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
		//z = -1;

		else if(result == startingLoc - sqzDIM){ //0,1
			++countResults;
		}

		//z =+1
		else if(result == startingLoc + sqzDIM){ //0,1
			++countResults;
		}

		else {
			qDebug() << result;
		}
	}
	 QCOMPARE(countResults,expectedResult);
}
