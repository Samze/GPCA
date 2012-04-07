
#include <QtCore/QCoreApplication>
#include <QtTest/QtTest>
#include "lattice2dtest.h"
#include "lattice3dtest.h"
#include "totalisitictest.h"
#include "generationstest.h"
#include "generations3dtest.h"
#include "outertotalistic3dtest.h"
#include "outertotalistictest.h"

int main(int argc, char *argv[])
{
	QCoreApplication a(argc, argv);
	
    QTest::qExec(&Lattice2DTest(), argc, argv);
    QTest::qExec(&Lattice3DTest(), argc, argv);
    QTest::qExec(&TotalisiticTest(), argc, argv);
    QTest::qExec(&GenerationsTest(), argc, argv);
    QTest::qExec(&Generations3DTest(), argc, argv);
	QTest::qExec(&OuterTotalisticTest(), argc, argv);
	QTest::qExec(&OuterTotalistic3DTest(), argc, argv);
	return a.exec();

}
