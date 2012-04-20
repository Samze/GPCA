#ifndef SCIARAPERFORMANCETEST_H
#define SCIARAPERFORMANCETEST_H

#include <QtTest/QtTest>
#include <QString>
#include "SCIARA.h"
#include "Lattice2D.h"
#include "CellularAutomata_GPGPU.h"
#include "CellularAutomata_CPU.h"

class SCIARAPerformanceTest : public QObject
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
	void testSCIARACPU();
	void testSCIARAGPU();



private:
	void testType(AbstractCellularAutomata* rule, Runtype runtype); 
};

#endif // PERFORMANCETEST_H
