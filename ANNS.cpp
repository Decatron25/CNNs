#include<bits/stdc++.h>
#include<random>
#include<time.h>
#include<cstdlib>
using namespace std;

class Neuron {
    int output;
    vector<double> weights;
};

class NueralNet {
    private:
        vector<vector<Neuron>> Layers;
    
    public:
        NueralNet(vector<int> topology) {
            
        }

        void InitializeWeights() {

        }


};


int main () {
    vector<int> topology = {3, 2, 1};
    NeuralNet a = NueralNet(topology);

    return 0;
}