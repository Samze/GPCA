/*	GPCA - A Cellular Automata library powered by CUDA. 
Copyright (C) 2011  Sam Gunaratne University of Plymouth

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "AbstractLattice.h"

AbstractLattice::AbstractLattice(){

}

AbstractLattice::AbstractLattice(unsigned int xDIM) : xDIM(xDIM) {

}

AbstractLattice::AbstractLattice(unsigned int xDIM, void* grid) :  xDIM(xDIM), pFlatGrid(grid) {

}

AbstractLattice::~AbstractLattice(void){
	delete[] pFlatGrid;
}

void AbstractLattice::setMaxBits(unsigned int newMaxBits) {

	maxBits = newMaxBits;
}

void AbstractLattice::setNoBits(unsigned int newNoBits) {

	noBits = newNoBits;
}

void AbstractLattice::setNeighbourhoodType(int ntype) {
	neighbourhoodType = ntype;
}

void AbstractLattice::setNoElements(unsigned int eleNo){
	noElements = eleNo;
}

void AbstractLattice::setXSize(unsigned int xSize){
	xDIM = xSize;
}