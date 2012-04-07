#ifndef GENERATIONS3DTEST_H
#define GENERATIONS3DTEST_H

#include <QtTest/QtTest>
#include <QString>
#include "Generations3D.h"
#include "Lattice3D.h"
#include "CellularAutomata_GPGPU.h"
#include "CellularAutomata_CPU.h"

class Generations3DTest : public QObject
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
	Generations3D* genGPU;
	Generations3D* genCPU;
	int xSize;
	int ySize;
	int zSize;

	bool arrayEquals(unsigned int* grid, int size, int expectedVal); 
};

#endif // GENERATIONS3DTEST_H
