#ifndef TOTALISITICTEST_H
#define TOTALISITICTEST_H

#include <QtTest/QtTest>
#include <QString>
#include "Totalistic.h"
#include "Generations.h"
#include "Lattice2D.h"

class TotalisiticTest : public QObject
{
	Q_OBJECT
private slots:
	void setStates();
};

#endif // TOTALISITICTEST_H
