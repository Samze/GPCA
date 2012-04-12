#include "SCIARA.h"


SCIARA::SCIARA(void)
{
	lattice = NULL;
}


SCIARA::~SCIARA(void)
{
}


map<void**, size_t>* SCIARA::getDynamicArrays() {

	map<void**, size_t>* newMap = new map<void**, size_t>();

	size_t gridMemSize = lattice->getXSize() * lattice->yDIM * sizeof(Cell);

	newMap->insert(make_pair((void**)&lattice->pFlatGrid, gridMemSize));
	//newMap->insert(make_pair((void**)&newGrid, gridMemSize));

	return newMap;
}

//TODO move this to .cpp
void SCIARA::setLattice(AbstractLattice* newLattice) {

	if(newLattice == lattice)
		return;
	
	if(lattice != NULL) {
		delete lattice;
	}
	lattice = NULL;

	Lattice2D* new2DLattice = dynamic_cast<Lattice2D*>(newLattice);

	lattice = new2DLattice;

	//newGrid = new SCIARA::Cell[lattice->xDIM * lattice->yDIM];
} 