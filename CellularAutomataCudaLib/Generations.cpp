#include "Generations.h"


Generations::Generations(void)
{
	lattice = NULL;
}


Generations::~Generations(void)
{
	delete lattice;
}

//These return a list of dynamic pointers to be put onto the GPU.
map<void**, size_t>* Generations::getDynamicArrays() {

		map<void**, size_t>* newMap = new map<void**, size_t>();

		size_t gridMemSize = lattice->getXSize() * lattice->yDIM * sizeof(unsigned int);

		newMap->insert(make_pair((void**)&lattice->pFlatGrid, gridMemSize));
		newMap->insert(make_pair((void**)&bornNo,sizeof(int) * bornSize));
		newMap->insert(make_pair((void**)&surviveNo,sizeof(int) * surviveSize));

		return newMap;
}

void Generations::setLattice(AbstractLattice* newLattice) {

	if(newLattice == lattice)
		return;

	if(lattice != NULL) {
		delete lattice;
	}
	lattice = NULL;

	Lattice2D* new2DLattice = dynamic_cast<Lattice2D*>(newLattice);

	lattice = new2DLattice;
} 