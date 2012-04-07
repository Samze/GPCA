#ifndef LATTICE3DTEST_H
#define LATTICE3DTEST_H

#include <QtTest/QtTest>
#include <QString>
#include "Lattice3D.h"

class Lattice3DTest : public QObject
{
	Q_OBJECT
		
private slots:
	void initTestCase();
    void cleanupTestCase();

	void Lattice3DConstructorOne();
	void Lattice3DConstructorTwo();

	void getMooresNeighbourhood();
	void getVonNeumannNeighbourhood();
	
private:
	Lattice3D* testLattice;
	int xDIM;
	int yDIM;
	int zDIM;
};

#endif // LATTICE3DTEST_H
