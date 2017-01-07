#ifndef BRAIN_H
#define BRAIN_H

#include <armadillo>

using namespace arma;

class Brain {

    int learningIterations;
    void imputeHiddenNode(umat*, umat*, double, mat);

  public:
    Brain(int);
    double learn(umat, umat);

};

#endif
