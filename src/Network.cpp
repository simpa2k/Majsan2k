#include <armadillo>

#include "headers/Network.h"
#include "utilities.cpp"

using namespace arma;

Network::Network() {

    thetaHidden = 0.75;
    thetaVisible = { {0.55, 0.95},
                     {0.60, 0.95},
                     {0.24, 0.42},
                     {0.13, 0.72},
                     {0.62, 0.66} };
    
    dataHidden = NULL;
    dataVisible = NULL;

}

double Network::getThetaHidden() {
    return thetaHidden;
}

mat Network::getThetaVisible() {
    return thetaVisible;
}

umat* Network::getDataHidden() {
    return dataHidden;
}

umat* Network::getDataVisible() {
    return dataVisible;
}

void Network::appendRows(umat* target, umat* addon) {
    target->insert_rows(target->n_rows, *addon);
}

void Network::appendCols(umat* target, umat* addon) {
    target->insert_cols(target->n_cols, *addon);
}

umat Network::simulateHiddenData(const int samples) {

    mat random = mat(1, samples, fill::randu);
    umat dataHidden = thetaHidden > random;

    return dataHidden;

}

umat Network::simulateVisibleData(umat dataHidden, const int samples) {

    mat colZero = trans(thetaVisible.col(0));
    mat colOne = trans(thetaVisible.col(1));

    expandHorizontally(&dataHidden, colZero.n_cols);

    expandVertically(&colZero, samples);
    expandVertically(&colOne, samples);

    umat dataVisibleProbFalse = ( (1 - dataHidden) % colZero ) > mat(samples, colZero.n_cols, fill::randu);
    umat dataVisibleProbTrue = ( dataHidden % colOne ) > mat(samples, colOne.n_cols, fill::randu);

    umat dataVisible = dataVisibleProbFalse + dataVisibleProbTrue;

    return dataVisible;

}

void Network::update(umat* dataHidden, umat* dataVisible) {

    if(!this->dataHidden) {
        this->dataHidden = dataHidden;
    } else {
        appendCols(this->dataHidden, dataHidden);
    }

    if(!this->dataVisible) {
        this->dataVisible = dataVisible;
    } else {
        appendRows(this->dataVisible, dataVisible);
    }

    thetaHidden = computeThetaHidden(this->dataHidden);
    thetaVisible = computeThetaVisible(this->dataHidden, this->dataVisible);

}
