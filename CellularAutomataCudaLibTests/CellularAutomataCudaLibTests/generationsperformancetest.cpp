#include "GenerationsPerformanceTest.h"

void GenerationsPerformanceTest::testGenerations2DCPU200() { testGens(CPU,TWOD,200);}
void GenerationsPerformanceTest::testGenerations2DGPU200() { testGens(GPU,TWOD,200);}
void GenerationsPerformanceTest::testGenerations3DCPU20() { testGens(CPU,THREED,20);}
void GenerationsPerformanceTest::testGenerations3DGPU20() { testGens(GPU,THREED,20);}

void GenerationsPerformanceTest::testGenerations2DCPU400(){ testGens(CPU,TWOD,400);}
void GenerationsPerformanceTest::testGenerations2DGPU400(){ testGens(GPU,TWOD,400);}
void GenerationsPerformanceTest::testGenerations3DCPU40(){ testGens(CPU,THREED,40);}
void GenerationsPerformanceTest::testGenerations3DGPU40(){ testGens(GPU,THREED,40);}


void GenerationsPerformanceTest::testGenerations2DCPU600(){ testGens(CPU,TWOD,600);}
void GenerationsPerformanceTest::testGenerations2DGPU600(){ testGens(GPU,TWOD,600);}
void GenerationsPerformanceTest::testGenerations3DCPU60(){ testGens(CPU,THREED,60);}
void GenerationsPerformanceTest::testGenerations3DGPU60(){ testGens(GPU,THREED,60);}


void GenerationsPerformanceTest::testGenerations2DCPU800(){ testGens(CPU,TWOD,800);}
void GenerationsPerformanceTest::testGenerations2DGPU800(){ testGens(GPU,TWOD,800);}
void GenerationsPerformanceTest::testGenerations3DCPU80(){ testGens(CPU,THREED,80);}
void GenerationsPerformanceTest::testGenerations3DGPU80(){ testGens(GPU,THREED,80);}


void GenerationsPerformanceTest::testGenerations2DCPU1000(){ testGens(CPU,TWOD,1000);}
void GenerationsPerformanceTest::testGenerations2DGPU1000(){ testGens(GPU,TWOD,1000);}
void GenerationsPerformanceTest::testGenerations3DCPU100(){ testGens(CPU,THREED,100);}
void GenerationsPerformanceTest::testGenerations3DGPU100(){ testGens(GPU,THREED,100);}

void GenerationsPerformanceTest::testGenerations2DCPU1200(){ testGens(CPU,TWOD,1200);}
void GenerationsPerformanceTest::testGenerations2DGPU1200(){ testGens(GPU,TWOD,1200);}
void GenerationsPerformanceTest::testGenerations3DCPU120(){ testGens(CPU,THREED,120);}
void GenerationsPerformanceTest::testGenerations3DGPU120(){ testGens(GPU,THREED,120);}

void GenerationsPerformanceTest::testGenerations2DCPU1400(){ testGens(CPU,TWOD,1400);}
void GenerationsPerformanceTest::testGenerations2DGPU1400(){ testGens(GPU,TWOD,1400);}
void GenerationsPerformanceTest::testGenerations3DCPU140(){ testGens(CPU,THREED,140);}
void GenerationsPerformanceTest::testGenerations3DGPU140(){ testGens(GPU,THREED,140);}

void GenerationsPerformanceTest::testGenerations2DCPU1600(){ testGens(CPU,TWOD,1600);}
void GenerationsPerformanceTest::testGenerations2DGPU1600(){ testGens(GPU,TWOD,1600);}
void GenerationsPerformanceTest::testGenerations3DCPU160(){ testGens(CPU,THREED,160);}
void GenerationsPerformanceTest::testGenerations3DGPU160(){ testGens(GPU,THREED,160);}

void GenerationsPerformanceTest::testGenerations2DCPU1800(){ testGens(CPU,TWOD,1800);}
void GenerationsPerformanceTest::testGenerations2DGPU1800(){ testGens(GPU,TWOD,1800);}
void GenerationsPerformanceTest::testGenerations3DCPU180(){ testGens(CPU,THREED,180);}
void GenerationsPerformanceTest::testGenerations3DGPU180(){ testGens(GPU,THREED,180);}

void GenerationsPerformanceTest::testGenerations2DCPU2000(){ testGens(CPU,TWOD,2000);}
void GenerationsPerformanceTest::testGenerations2DGPU2000(){ testGens(GPU,TWOD,2000);}
void GenerationsPerformanceTest::testGenerations3DCPU200(){ testGens(CPU,THREED,200);}
void GenerationsPerformanceTest::testGenerations3DGPU200(){ testGens(GPU,THREED,200);}


	
void GenerationsPerformanceTest::testGens(Runtype type, Dimension d, int size){

	Totalistic* gen;
	int* bornNumOne = new int[1]; 
	bornNumOne[0] = 0;

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

		//QBENCHMARK {
			testType(gen,type, size);
	//	}
	
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
	
	int totalTime = 0;
	
	const int iterations = 1;

	for(int i = 0; i < iterations; i++) {
		QTime t;
		t.start();
		run->nextTimeStep();
		totalTime += t.elapsed();
	}

	float avg  = ((float)totalTime / iterations);

	qDebug("T = %3.3f",avg);


	//DELETE run, this will break the test
}