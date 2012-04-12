#include "SCIARAThickness.h"


SCIARAThickness::SCIARAThickness(void)
{
	lattice = NULL;
}


SCIARAThickness::~SCIARAThickness(void)
{
	delete lattice;
}


map<void**, size_t>* SCIARAThickness::getDynamicArrays() {

	map<void**, size_t>* newMap = new map<void**, size_t>();

	size_t gridMemSize = lattice->getXSize() * lattice->yDIM * sizeof(Cell);

	newMap->insert(make_pair((void**)&lattice->pFlatGrid, gridMemSize));
	//newMap->insert(make_pair((void**)&newGrid, gridMemSize));

	return newMap;
}


//TODO move this to .cpp
void SCIARAThickness::setLattice(AbstractLattice* newLattice) {

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