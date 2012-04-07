#ifndef OUTERTOTALISTIC3DTEST_H
#define OUTERTOTALISTIC3DTEST_H

#include <QtTest/QtTest>
#include <QString>
#include "OuterTotalistic3D.h"
#include "Lattice3D.h"
#include "CellularAutomata_GPGPU.h"
#include "CellularAutomata_CPU.h"

class OuterTotalistic3DTest : public QObject
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
	OuterTotalistic3D* genGPU;
	OuterTotalistic3D* genCPU;
	int xSize;
	int ySize;
	int zSize;

	bool arrayEquals(unsigned int* grid, int size, int expectedVal); 
};


#endif // OUTERTOTALISTIC3DTEST_H
