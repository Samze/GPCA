#ifndef LATTICE2DTEST_H
#define LATTICE2DTEST_H

#include <QtTest/QtTest>
#include <QString>
#include "Lattice2D.h"

class Lattice2DTest : public QObject
{
	Q_OBJECT
		
private slots:
	void initTestCase();
    void cleanupTestCase();

	void Lattice2DConstructorOne();
	void Lattice2DConstructorTwo();

	void getMooresNeighbourhood();
	void getVonNeumannNeighbourhood();
	
private:
	Lattice2D* testLattice;
	int xDIM;
	int yDIM;
};

#endif // LATTICE2DTEST_H
