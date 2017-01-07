/*
 * Class representing a Bayesian Network. Currently only serves as the
 * model where data is stored and true probabilities based on actual,
 * observed data are recorded.
 *
 * Also contains methods to simulate data based on the probabilites recorded.
 * These are simply translations of functions from Jace Kohlmeier's Bayesian 
 * network example written in python and all credit for those functions goes to him.
 * The article can be found at http://derandomized.com/post/20009997725/bayes-net-example-with-python-and-khanacademy.
 */

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
