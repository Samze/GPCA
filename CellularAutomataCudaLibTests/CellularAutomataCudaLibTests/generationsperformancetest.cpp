#include "GenerationsPerformanceTest.h"

//void GenerationsPerformanceTest::testGenerations2DCPU20() { testGens(CPU,TWOD,20);}
//void GenerationsPerformanceTest::testGenerations2DGPU20() { testGens(GPU,TWOD,20);}
//void GenerationsPerformanceTest::testGenerations3DCPU20() { testGens(CPU,THREED,20);}
//void GenerationsPerformanceTest::testGenerations3DGPU20() { testGens(GPU,THREED,20);}
//
//void GenerationsPerformanceTest::testGenerations2DCPU40(){ testGens(CPU,TWOD,40);}
//void GenerationsPerformanceTest::testGenerations2DGPU40(){ testGens(GPU,TWOD,40);}
//void GenerationsPerformanceTest::testGenerations3DCPU40(){ testGens(CPU,THREED,40);}
//void GenerationsPerformanceTest::testGenerations3DGPU40(){ testGens(GPU,THREED,40);}
//
//
//void GenerationsPerformanceTest::testGenerations2DCPU60(){ testGens(CPU,TWOD,60);}
//void GenerationsPerformanceTest::testGenerations2DGPU60(){ testGens(GPU,TWOD,60);}
//void GenerationsPerformanceTest::testGenerations3DCPU60(){ testGens(CPU,THREED,60);}
//void GenerationsPerformanceTest::testGenerations3DGPU60(){ testGens(GPU,THREED,60);}
//
//
//void GenerationsPerformanceTest::testGenerations2DCPU80(){ testGens(CPU,TWOD,80);}
//void GenerationsPerformanceTest::testGenerations2DGPU80(){ testGens(GPU,TWOD,80);}
//void GenerationsPerformanceTest::testGenerations3DCPU80(){ testGens(CPU,THREED,80);}
//void GenerationsPerformanceTest::testGenerations3DGPU80(){ testGens(GPU,THREED,80);}
//
//
//void GenerationsPerformanceTest::testGenerations2DCPU100(){ testGens(CPU,TWOD,100);}
//void GenerationsPerformanceTest::testGenerations2DGPU100(){ testGens(GPU,TWOD,100);}
//void GenerationsPerformanceTest::testGenerations3DCPU100(){ testGens(CPU,THREED,100);}
//void GenerationsPerformanceTest::testGenerations3DGPU100(){ testGens(GPU,THREED,100);}
//
//void GenerationsPerformanceTest::testGenerations2DCPU200(){ testGens(CPU,TWOD,200);}
//void GenerationsPerformanceTest::testGenerations2DGPU200(){ testGens(GPU,TWOD,200);}
//void GenerationsPerformanceTest::testGenerations3DCPU200(){ testGens(CPU,THREED,200);}
//void GenerationsPerformanceTest::testGenerations3DGPU200(){ testGens(GPU,THREED,200);}
//
//void GenerationsPerformanceTest::testGenerations2DCPU400(){ testGens(CPU,TWOD,400);}
//void GenerationsPerformanceTest::testGenerations2DGPU400(){ testGens(GPU,TWOD,400);}
//
//void GenerationsPerformanceTest::testGenerations2DCPU600(){ testGens(CPU,TWOD,600);}
void GenerationsPerformanceTest::testGenerations2DGPU600(){ testGens(GPU,TWOD,20);}
//
//void GenerationsPerformanceTest::testGenerations2DCPU800(){ testGens(CPU,TWOD,800);}
//void GenerationsPerformanceTest::testGenerations2DGPU800(){ testGens(GPU,TWOD,800);}
//
//void GenerationsPerformanceTest::testGenerations2DCPU1000(){ testGens(CPU,TWOD,1000);}
//void GenerationsPerformanceTest::testGenerations2DGPU1000(){ testGens(GPU,TWOD,1000);}


	
void GenerationsPerformanceTest::testGens(Runtype type, Dimension d, int size){

	Totalistic* gen;
	int* bornNumOne = new int[1]; 

	if(d == TWOD) {
		gen = new Generations();
		gen->setBornNo(bornNumOne,1);
		AbstractLattice* lattice = new Lattice2D(size,size,1);
		lattice->setNeighbourhoodType(Lattice2D::MOORE);
		gen->setLattice(lattice);
		gen->setStates(20);
	}
	else {
		gen = new Generations3D();
		gen->setBornNo(bornNumOne,1);
		
		AbstractLattice* lattice= new Lattice3D(size,size,size,1);
		lattice->setNeighbourhoodType(Lattice3D::MOORE_3D);
		gen->setLattice(lattice);
		gen->setStates(20);
	}


		QBENCHMARK {
			testType(gen,type, size);
		}
	delete gen;
}



void GenerationsPerformanceTest::testType(AbstractCellularAutomata* rule, Runtype runtype, int size) {
	
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