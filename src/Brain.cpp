#include <armadillo>

#include "headers/Brain.h"
#include "utilities.cpp"

using namespace std;
using namespace arma;

Brain::Brain(int learningIterations) : learningIterations(learningIterations) {};

mat Brain::imputeHiddenNode(umat* dataVisible, double thetaHidden, mat thetaVisible, bool generateNewData) {
    
    mat colZero = trans(thetaVisible.col(0));
    mat colOne = trans(thetaVisible.col(1));

    expandVertically(&colZero, dataVisible->n_rows);
    expandVertically(&colOne, dataVisible->n_rows);

    mat probVis0 = colZero % *dataVisible + (1 - colZero) % (1 - *dataVisible);
    mat probVis0Unnorm = (1 - thetaHidden) * prod(probVis0, 1);

    mat probVis1 = colOne % *dataVisible + (1 - colOne) % (1 - *dataVisible);
    mat probVis1Unnorm = thetaHidden * prod(probVis1, 1);

    mat hidden = probVis1Unnorm / (probVis0Unnorm + probVis1Unnorm);
    hidden.transform( [] (double val) { return (isnan(val) ? double(0) : val); });

    if (generateNewData) {
        hidden = conv_to<mat>::from(trans(hidden > mat(hidden.n_rows, hidden.n_cols, fill::randu)));
    } 

    return hidden;

}

double Brain::learn(umat dataHidden, umat dataVisible) {
    
    double thetaHidden = computeThetaHidden(&dataHidden);
    mat thetaVisible = computeThetaVisible(&dataHidden, &dataVisible);

    for (int i = 0; i < learningIterations; ++i) {
       
       dataHidden = conv_to<umat>::from(imputeHiddenNode(&dataVisible, thetaHidden, thetaVisible, true)); 

       if (computeThetaHidden(&dataHidden) < 0.5) {
           dataHidden = 1 - dataHidden;
       }

       thetaHidden = computeThetaHidden(&dataHidden);
       thetaVisible = computeThetaVisible(&dataHidden, &dataVisible);

    }
    return thetaHidden;
}
