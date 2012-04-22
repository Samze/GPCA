#ifndef PERFORMANCETEST_H
#define PERFORMANCETEST_H

#include <QtTest/QtTest>
#include <QString>
#include "OuterTotalistic.h"
#include "Lattice2D.h"
#include "CellularAutomata_GPGPU.h"
#include "CellularAutomata_CPU.h"

class GenerationsPerformanceTest : public QObject
{
	Q_OBJECT

	enum Runtype {
		CPU,
		GPU
	};

	enum Dimension {
		TWOD,
		THREED
	};

private slots:
	void testGenerations2DCPU200();
	void testGenerations2DGPU200();
	void testGenerations3DCPU20();
	void testGenerations3DGPU20();

	void testGenerations2DCPU400();
	void testGenerations2DGPU400();
	void testGenerations3DCPU40();
	void testGenerations3DGPU40();

	void testGenerations2DCPU600();
	void testGenerations2DGPU600();
	void testGenerations3DCPU60();
	void testGenerations3DGPU60();

	void testGenerations2DCPU800();
	void testGenerations2DGPU800();
	void testGenerations3DCPU80();
	void testGenerations3DGPU80();

	void testGenerations2DCPU1000();
	void testGenerations2DGPU1000();
	void testGenerations3DCPU100();
	void testGenerations3DGPU100();

	void testGenerations2DCPU1200();
	void testGenerations2DGPU1200();
	void testGenerations3DCPU120();
	void testGenerations3DGPU120();

	void testGenerations2DCPU1400();
	void testGenerations2DGPU1400();
	void testGenerations3DCPU140();
	void testGenerations3DGPU140();

	void testGenerations2DCPU1600();
	void testGenerations2DGPU1600();
	void testGenerations3DCPU160();
	void testGenerations3DGPU160();

	void testGenerations2DCPU1800();
	void testGenerations2DGPU1800();
	void testGenerations3DCPU180();
	void testGenerations3DGPU180();

	void testGenerations2DCPU2000();
	void testGenerations2DGPU2000();
	void testGenerations3DCPU200();
	void testGenerations3DGPU200();

private:
	void testGens(Runtype runtype, Dimension d, int size);
	void testType(AbstractCellularAutomata* rule, Runtype runtype, int size); 
};

#endif // PERFORMANCETEST_H
