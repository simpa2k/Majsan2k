#ifndef NETWORK_H
#define NETWORK_H

#include <armadillo>

using namespace arma;

class Network {

    double thetaHidden;
    mat thetaVisible;

    umat* dataHidden;
    umat* dataVisible;

    void appendRows(umat*, umat*);
    void appendCols(umat*, umat*);

  public:
    Network();

    double getThetaHidden();  
    mat getThetaVisible();

    umat* getDataHidden();  
    umat* getDataVisible();

    //umat simulateHiddenData(const double*, const int);
    umat simulateHiddenData(const int);
    //umat simulateVisibleData(umat, const mat*, const int);
    umat simulateVisibleData(umat, const int);

    void update(umat*, umat*);

};

#endif
