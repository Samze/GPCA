#ifndef OUTERPERFORMANCETEST_H
#define OUTERPERFORMANCETEST_H

#include <QtTest/QtTest>
#include <QString>
#include "OuterTotalistic.h"
#include "Lattice2D.h"
#include "CellularAutomata_GPGPU.h"
#include "CellularAutomata_CPU.h"

class OuterTotPerformanceTest : public QObject
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
	void testOuterTot2DCPU20();
	void testOuterTot2DGPU20();
	void testOuterTot3DCPU20();
	void testOuterTot3DGPU20();

	void testOuterTot2DCPU40();
	void testOuterTot2DGPU40();
	void testOuterTot3DCPU40();
	void testOuterTot3DGPU40();


	void testOuterTot2DCPU60();
	void testOuterTot2DGPU60();
	void testOuterTot3DCPU60();
	void testOuterTot3DGPU60();


	void testOuterTot2DCPU80();
	void testOuterTot2DGPU80();
	void testOuterTot3DCPU80();
	void testOuterTot3DGPU80();


	void testOuterTot2DCPU100();
	void testOuterTot2DGPU100();
	void testOuterTot3DCPU100();
	void testOuterTot3DGPU100();

	void testOuterTot2DCPU200();
	void testOuterTot2DGPU200();
	void testOuterTot3DCPU200();
	void testOuterTot3DGPU200();

	void testOuterTot2DCPU400();
	void testOuterTot2DGPU400();

private:
	void testGens(Runtype runtype, Dimension d, int size);
	void testType(AbstractCellularAutomata* rule, Runtype runtype, int size); 
};

#endif // PERFORMANCETEST_H
