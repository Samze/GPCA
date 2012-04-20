#include "OuterTotPerformanceTest.h"

void OuterTotPerformanceTest::testOuterTot2DCPU20() { testGens(CPU,TWOD,20);}
void OuterTotPerformanceTest::testOuterTot2DGPU20() { testGens(GPU,TWOD,20);}
void OuterTotPerformanceTest::testOuterTot3DCPU20() { testGens(CPU,THREED,20);}
void OuterTotPerformanceTest::testOuterTot3DGPU20() { testGens(GPU,THREED,20);}

void OuterTotPerformanceTest::testOuterTot2DCPU40(){ testGens(CPU,TWOD,40);}
void OuterTotPerformanceTest::testOuterTot2DGPU40(){ testGens(GPU,TWOD,40);}
void OuterTotPerformanceTest::testOuterTot3DCPU40(){ testGens(CPU,THREED,40);}
void OuterTotPerformanceTest::testOuterTot3DGPU40(){ testGens(GPU,THREED,40);}


void OuterTotPerformanceTest::testOuterTot2DCPU60(){ testGens(CPU,TWOD,60);}
void OuterTotPerformanceTest::testOuterTot2DGPU60(){ testGens(GPU,TWOD,60);}
void OuterTotPerformanceTest::testOuterTot3DCPU60(){ testGens(CPU,THREED,60);}
void OuterTotPerformanceTest::testOuterTot3DGPU60(){ testGens(GPU,THREED,60);}


void OuterTotPerformanceTest::testOuterTot2DCPU80(){ testGens(CPU,TWOD,80);}
void OuterTotPerformanceTest::testOuterTot2DGPU80(){ testGens(GPU,TWOD,80);}
void OuterTotPerformanceTest::testOuterTot3DCPU80(){ testGens(CPU,THREED,80);}
void OuterTotPerformanceTest::testOuterTot3DGPU80(){ testGens(GPU,THREED,80);}


void OuterTotPerformanceTest::testOuterTot2DCPU100(){ testGens(CPU,TWOD,100);}
void OuterTotPerformanceTest::testOuterTot2DGPU100(){ testGens(GPU,TWOD,100);}
void OuterTotPerformanceTest::testOuterTot3DCPU100(){ testGens(CPU,THREED,100);}
void OuterTotPerformanceTest::testOuterTot3DGPU100(){ testGens(GPU,THREED,100);}

void OuterTotPerformanceTest::testOuterTot2DCPU200(){ testGens(CPU,TWOD,200);}
void OuterTotPerformanceTest::testOuterTot2DGPU200(){ testGens(GPU,TWOD,200);}
void OuterTotPerformanceTest::testOuterTot3DCPU200(){ testGens(CPU,THREED,200);}
void OuterTotPerformanceTest::testOuterTot3DGPU200(){ testGens(GPU,THREED,200);}

void OuterTotPerformanceTest::testOuterTot2DCPU400(){ testGens(CPU,TWOD,400);}
void OuterTotPerformanceTest::testOuterTot2DGPU400(){ testGens(GPU,TWOD,400);}


	
void OuterTotPerformanceTest::testGens(Runtype type, Dimension d, int size){

	Totalistic* gen;
	int* bornNumOne = new int[1]; 

	if(d == TWOD) {
		gen = new OuterTotalistic();
		gen->setBornNo(bornNumOne,1);
		AbstractLattice* lattice = new Lattice2D(size,size,1);
		lattice->setNeighbourhoodType(Lattice2D::MOORE);
		gen->setLattice(lattice);
		gen->setStates(2);
	}
	else {
		gen = new OuterTotalistic3D();
		gen->setBornNo(bornNumOne,1);
		
		AbstractLattice* lattice= new Lattice3D(size,size,size,1);
		lattice->setNeighbourhoodType(Lattice3D::MOORE_3D);
		gen->setLattice(lattice);
		gen->setStates(2);
	}


		QBENCHMARK {
			testType(gen,type, size);
		}
	delete gen;
}



void OuterTotPerformanceTest::testType(AbstractCellularAutomata* rule, Runtype runtype, int size) {
	
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