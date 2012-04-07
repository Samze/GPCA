#include "totalisitictest.h"
#include "Lattice2D.h"

void TotalisiticTest::setStates(){

	Generations* genTest = new Generations();
	Lattice2D* newlattice = new Lattice2D(10,10,1);

	genTest->setLattice(newlattice);
	AbstractLattice* lattice = genTest->getLattice();

	int stateSize = 2;
	genTest->setStates(stateSize);
	
	QCOMPARE(genTest->getNoStates(),stateSize);
	QCOMPARE(lattice->getNoBits(),1);
	QCOMPARE((int)lattice->getMaxBits(),1);


	stateSize = 4;
	genTest->setStates(stateSize);
	
	QCOMPARE(genTest->getNoStates(),stateSize);
	QCOMPARE(lattice->getNoBits(),2);
	QCOMPARE((int)lattice->getMaxBits(),3);

	
	stateSize = 101;
	genTest->setStates(stateSize);
	
	QCOMPARE(genTest->getNoStates(),stateSize);
	QCOMPARE(lattice->getNoBits(),7);
	QCOMPARE((int)lattice->getMaxBits(),127);

}