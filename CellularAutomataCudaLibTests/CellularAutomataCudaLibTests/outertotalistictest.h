#ifndef OUTERTOTALISITICTEST_H
#define OUTERTOTALISITICTEST_H


#include <QtTest/QtTest>
#include <QString>
#include "OuterTotalistic.h"
#include "Lattice2D.h"
#include "CellularAutomata_GPGPU.h"
#include "CellularAutomata_CPU.h"

class OuterTotalisticTest :  public QObject
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
	OuterTotalistic* genGPU;
	OuterTotalistic* genCPU;
	int xSize;
	int ySize;

	bool arrayEquals(unsigned int* grid, int size, int expectedVal); 
};

#endif // OUTERTOTALISITICTEST_H
