/*
 * Class representing a brain that can calculate the probability of
 * a hidden node being activated given the correct data and recorded
 * probabilities based on observations. 
 *
 * At the moment, this code is simply a translation from a python
 * version written by Jace Kohlmeier, presented at http://derandomized.com/post/20009997725/bayes-net-example-with-python-and-khanacademy. 
 * All credit goes to him.
 */

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
