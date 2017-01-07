#include <iostream>
#include <stdlib.h>
#include <armadillo>

#include "Brain.cpp"
#include "Network.cpp"

using namespace std;
using namespace arma;

int main() {

    srand(time(0));
    arma_rng::set_seed_random();

    int rowcount = 10000;

    Network network = Network();

    double thetaHidden = network.getThetaHidden();
    mat thetaVisible = network.getThetaVisible();

    //umat dataHidden = network.simulateHiddenData(&thetaHidden, rowcount);
    umat dataHidden = network.simulateHiddenData(rowcount);
    //umat dataVisible = network.simulateVisibleData(trans(dataHidden), &thetaVisible, rowcount);
    umat dataVisible = network.simulateVisibleData(trans(dataHidden), rowcount);

    dataHidden.imbue( [&]() { return rand() % 2; } ); 

    Brain brain = Brain(400);
    double thetaLearned = brain.learn(dataHidden, dataVisible);

    cout << "True probability of hidden node being active: " << thetaHidden << endl;
    cout << "Estimated probability of hidden node being active: " << thetaLearned << endl;

    return 0;

}
