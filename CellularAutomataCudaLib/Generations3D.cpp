#include "Generations3D.h"


Generations3D::Generations3D(void)
{
	lattice = NULL;
}


Generations3D::~Generations3D(void)
{
	delete lattice;
}

//These return a list of dynamic pointers to be put onto the GPU.
map<void**, size_t>* Generations3D::getDynamicArrays() {

	map<void**, size_t>* newMap = new map<void**, size_t>();

	size_t gridMemSize = lattice->getXSize() * lattice->yDIM * lattice->zDIM * sizeof(unsigned int);

	newMap->insert(make_pair((void**)&lattice->pFlatGrid, gridMemSize));
	newMap->insert(make_pair((void**)&bornNo,sizeof(int) * bornSize));
	newMap->insert(make_pair((void**)&surviveNo,sizeof(int) * surviveSize));

	return newMap;
}

void Generations3D::setLattice(AbstractLattice* newLattice) {

	if(newLattice == lattice)
		return;

	if(lattice != NULL) {
		delete lattice;
	}

	lattice = NULL;

	Lattice3D* new3DLattice = dynamic_cast<Lattice3D*>(newLattice);

	lattice = new3DLattice;
} 
