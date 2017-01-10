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

umat* gatherVisibleData(Network* network) {
    
    umat* visibleData = new umat(1, 5);
    visibleData->imbue( []() { return rand() % 2; } );

    network->updateVisible(visibleData);

    return visibleData;

}

int gatherHiddenData(Network* network) {

    umat* hiddenData = new umat(1, 1);

    int random = rand() % 2;
    hiddenData->imbue( [=]() { return random; } );

    network->updateHidden(hiddenData);

    return random;
    
}

void realisticRun() {

    Network network = Network();
    Brain brain = Brain(100);

    for (int i = 0; i < 5; ++i) {
        gatherHiddenData(&network);
        gatherVisibleData(&network);
    }

    cout << network.getThetaHidden() << endl;

    double thetaLearned;
    double correctGuesses = 0;

    for (int i = 0; i < 1000; ++i) {
        
        umat* visibleData = gatherVisibleData(&network);

        umat* tempHidden = new umat(1, 1);
        tempHidden->imbue( []() { return rand() % 2; } );

        umat* hiddenData = network.getDataHidden();
        tempHidden->insert_cols(tempHidden->n_rows, *hiddenData);

        thetaLearned = brain.learn(*tempHidden, *network.getDataVisible());

        int guess = thetaLearned > 0.5 ? 1 : 0;
        /*if (thetaLearned > 0.5) {
            cout << "Next hidden will be 1." << endl;
        } else if (thetaLearned < 0.5) {
            cout << "Next hidden will be 0." << endl;
        } else {
            cout << "Both 1 and 0 equally likely." << endl;
        }*/

        int result = computeThetaHidden(visibleData) < 0.3 ? 1 : 0;

        umat* resultingData = new umat(1, 1);
        resultingData->fill(result);
        network.updateHidden(resultingData);

        if (guess == result) {
            correctGuesses++;
        }
        
    }
    cout << thetaLearned << endl;
    cout << correctGuesses << endl;
}

int main() {

    srand(time(0));
    arma_rng::set_seed_random();

    realisticRun();
    //simulatedRun();

    return 0;

}
