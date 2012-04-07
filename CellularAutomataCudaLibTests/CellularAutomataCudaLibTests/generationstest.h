#ifndef GENERATIONSTEST_H
#define GENERATIONSTEST_H

#include <QtTest/QtTest>
#include <QString>
#include "Generations.h"
#include "Lattice2D.h"
#include "CellularAutomata_GPGPU.h"
#include "CellularAutomata_CPU.h"

class GenerationsTest : public QObject
{
	Q_OBJECT
private slots:
	void initTestCase();
    void cleanupTestCase();

	void performanceCPU();
	void performanceGPU();
	void iterations();

private:
	CellularAutomata* CAGPU;
	CellularAutomata* CACPU;
	Generations* genGPU;
	Generations* genCPU;
	int xSize;
	int ySize;

	bool arrayEquals(unsigned int* grid, int size, int expectedVal); 
};
#endif // LAUNCHGENERATIONSTEST_H
