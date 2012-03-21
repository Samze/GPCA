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

#include "CellularAutomata.h"


CellularAutomata::CellularAutomata() {

}


CellularAutomata::~CellularAutomata() {

	delete caRule;

	caRule = NULL;
}


DLLExport void CellularAutomata::setCARule(AbstractCellularAutomata* ca) {
	caRule = ca;

	Generations* v = dynamic_cast<Generations*>(caRule);
	OuterTotalistic* v2 = dynamic_cast<OuterTotalistic*>(caRule);
	Generations3D* v3 = dynamic_cast<Generations3D*>(caRule);
	SCIARA* v4 = dynamic_cast<SCIARA*>(caRule);
	SCIARA2* v5 = dynamic_cast<SCIARA2*>(caRule);
	
	//No support for Runtime polymorphism inside the kernel.
	if(v != 0) {
		ruleName = "Generations";
	}
	else if (v2 != 0) {
		ruleName = "OuterTotalistic";
	}
	else if(v3 != 0) {
		ruleName = "Generations";
	}
	else if(v4 != 0) {
		ruleName = "SCIARA";
	}
	else if(v5 != 0) {
		ruleName = "SCIARA";
	}
}