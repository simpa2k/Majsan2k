#include <iostream>
#include <stdlib.h>
#include <armadillo>

#include "Brain.cpp"
#include "Network.cpp"

using namespace std;
using namespace arma;

void simulatedRun() {

    double thetaHidden = 0.75;
    mat thetaVisible = { {0.55, 0.95},
                         {0.60, 0.95},
                         {0.24, 0.42},
                         {0.13, 0.72},
                         {0.62, 0.66} };

    int rowcount = 10000;

    Network network = Network(thetaHidden, thetaVisible);

    thetaHidden = network.getThetaHidden();
    thetaVisible = network.getThetaVisible();

    umat dataHidden = network.simulateHiddenData(rowcount);
    umat dataVisible = network.simulateVisibleData(trans(dataHidden), rowcount);

    dataHidden.imbue( [&]() { return rand() % 2; } ); 

    Brain brain = Brain(400);
    double thetaLearned = brain.learn(dataHidden, dataVisible);

    cout << "True probability of hidden node being active: " << thetaHidden << endl;
    cout << "Estimated probability of hidden node being active: " << thetaLearned << endl;

}

umat gatherVisibleData(Network* network) {
    
    umat hiddenData = network->simulateHiddenData(1);
    umat visibleData = network->simulateVisibleData(hiddenData, 1);

    network->updateVisible(&visibleData);

    return visibleData;

}

umat gatherHiddenData(Network* network) {
    
    umat hiddenData = network->simulateHiddenData(1);
    network->updateHidden(&hiddenData);

    return hiddenData;

}


void realisticRun() {

    double thetaHidden = 0.75;
    mat thetaVisible = { {0.55, 0.95},
                         {0.60, 0.95},
                         {0.24, 0.42},
                         {0.13, 0.72},
                         {0.62, 0.66} };

    Network network = Network(thetaHidden, thetaVisible);
    Brain brain = Brain(400);

    umat hiddenData = network.simulateHiddenData(1000);
    umat visibleData = network.simulateVisibleData(trans(hiddenData), 1000);

    network.update(&hiddenData, &visibleData);
    
    double thetaLearned = brain.learn(*network.getDataHidden(), *network.getDataVisible());

    std::cout << thetaLearned << std::endl;
    std::cout << "Starting loop." << std::endl << std::endl;

    for (int i = 0; i < 10; ++i) {

        umat newVisibleData = gatherVisibleData(&network);

        mat guess = brain.imputeHiddenNode(&newVisibleData, network.getThetaHidden(), network.getThetaVisible(), false);

        std::cout << "Guess: ";
        guess.print();
        cout << std::endl;

        gatherHiddenData(&network);
        network.updateThetaVisible();

        mat actual = brain.imputeHiddenNode(&newVisibleData, network.getThetaHidden(), network.getThetaVisible(), false);

        std::cout << "Actual: ";
        actual.print(); 
        cout << std::endl;

    }

}

umat gatherPredictableVisibleData(Network* network, umat hiddenData) {

    umat* visibleData; 

    if (hiddenData(0, 0) == 1) {
       visibleData = new umat(1, 5, fill::ones);
    } else {
        visibleData = new umat(1, 5, fill::zeros);
    }

    network->updateVisible(visibleData);

    return *visibleData;

}

umat* gatherPredictableHiddenData() {
    
    umat* hiddenData = new umat(1, 1);
    hiddenData->imbue( []() { return rand() % 2; } );

    return hiddenData;
    
}

void gatherPredictableData(Network* network) {

    for (int i = 0; i < 5; ++i) {
        
        umat* hiddenData = gatherPredictableHiddenData();
        umat visibleData = gatherPredictableVisibleData(network, *hiddenData);

        network->updateHidden(hiddenData);

    }

    network->updateThetaVisible();

}

void predictableRun() {

    Network network = Network();
    Brain brain = Brain(400);
    
    gatherPredictableData(&network);

    for (int i = 0; i < 10; ++i) {

        umat* newHiddenData = gatherPredictableHiddenData();
        umat newVisibleData = gatherPredictableVisibleData(&network, *newHiddenData);

        mat guess = brain.imputeHiddenNode(&newVisibleData, network.getThetaHidden(), network.getThetaVisible(), false);

        std::cout << "Guess: ";
        guess.print();
        cout << std::endl;

        network.updateHidden(newHiddenData);
        network.updateThetaVisible();

        std::cout << "Actual: "; 
        newHiddenData->print();
        cout << endl;

    }
    
}

int main() {

    srand(time(0));
    arma_rng::set_seed_random();

    //realisticRun();
    //simulatedRun();
    predictableRun();

    return 0;

}
