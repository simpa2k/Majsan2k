#ifndef THETA_H
#define THETA_H

#include <armadillo>

using namespace arma;

double computeThetaHidden(umat*);
mat computeThetaVisible(umat*, umat*);
void expandVertically(mat*, int);
void expandHorizontally(umat*, int);

#endif
