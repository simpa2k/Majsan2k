#include <iostream>
#include <stdlib.h>
#include <armadillo>

#include "Brain.cpp"
#include "Network.cpp"

using namespace std;
using namespace arma;

void simulatedRun() {

    int rowcount = 10000;

    Network network = Network();

    double thetaHidden = network.getThetaHidden();
    mat thetaVisible = network.getThetaVisible();

    umat dataHidden = network.simulateHiddenData(rowcount);
    umat dataVisible = network.simulateVisibleData(trans(dataHidden), rowcount);

    dataHidden.imbue( [&]() { return rand() % 2; } ); 

    Brain brain = Brain(400);
    double thetaLearned = brain.learn(dataHidden, dataVisible);

    cout << "True probability of hidden node being active: " << thetaHidden << endl;
    cout << "Estimated probability of hidden node being active: " << thetaLearned << endl;

}

void realisticRun() {

    Network network = Network();
    Brain brain = Brain(400);

    for (int i = 0; int i < 5; ++i) {
        gatherVisibleData(network);
        gatherHiddenData(network);
    }

    double thetaLearned;

    for (int i = 0; int i < 100; ++i) {
        
        umat visibleData = gatherVisibleData(network);
        thetaLearned = brain.learn(*network.getDataHidden(), *network.getDataVisible());

        if (thetaLearned > 0.5) {
            cout << "Next hidden will be 1." << endl;
        } else if (thetaLearned < 0.5) {
            cout << "Next hidden will be 0." << endl;
        } else {
            cout << "Both 1 and 0 equally likely." << endl;
        }

        int result = computeThetaHidden(&visibleData) > 0.5 ? 1 : 0;

        umat hiddenData = umat(1, 1);
        hiddenData.fill(result);
        network.updateHidden(hiddenData);

        cout << "Result was: " << result << endl;
        
    }
}

umat gatherVisibleData(Network* network) {
    
    umat visibleData = mat(1, 5);
    visibleData.imbue( []() { return rand() % 2; } );

    network.updateVisible(visibleData);

    return visibleData;

}

int gatherHiddenData(Network* network) {

    umat hiddenData = mat(1, 1);

    int random = rand() % 2;
    hiddenData.imbue( []() { return random; } );

    network.updateHidden(hiddenData);

    return random;
    
}

int main() {

    srand(time(0));
    arma_rng::set_seed_random();


    return 0;

}
