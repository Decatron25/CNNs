#include<bits/stdc++.h>
#include<random>
#include<time.h>
#include<cstdlib>
#include<cmath>
using namespace std;



class Neuron {

public:
    double output;
    vector<double> weights;
    
    Neuron (double output) {
        this->output = output;
    }

    void initializeWeights(int weightSize, int n_index) {
        srand(time(0));
        weights.resize(weightSize);
        if (n_index == 0) {
            // double temp = (double)rand()/RAND_MAX;
            double temp =1;
            for(int i=0;i<weightSize;i++){
                this->weights[i] = temp;
            }
            return;
        }
            
        
        for(int i=0;i<weightSize;i++){
            // this->weights[i] = (double)rand()/RAND_MAX;
            this->weights[i] = 1;
        }
        
    }

    //n_index is the index of neuron in the layer
    void forwardPassFirstLayer(vector<double>& input, int n_index) {
        if (n_index == 0) {
            output = 1;
        } else {
            output = input[n_index - 1];
        } 

    }

    double sigmoidActivation (double x) {
         return 1.0 / (1.0 + exp(-1.0 * x));
    }

    void forwardPassSigmoid(vector<Neuron>& prevLayer, int n_index) {
        double sum = 0;
        for (int i = 0; i < prevLayer.size(); i++) {
            sum += prevLayer[i].output * prevLayer[i].weights[n_index];
        }
        output = sigmoidActivation(sum);
    }


    void forwardPassSoftmax(vector<Neuron>& prevLayer, int n_index) {
        double sum = 0;
        for (int i = 0; i < prevLayer.size(); i++) {
            sum += prevLayer[i].output * prevLayer[i].weights[n_index];
        }
    }


    
    void backwardPass() {

    }
};

typedef vector<Neuron> Layer;

class NeuralNet {
    
    public:
        vector<Layer>Layers;

        NeuralNet(vector<int> topology) {
            Layers.resize(topology.size());
            for(int i=0;i<topology.size();i++){
                Layers[i].push_back(Neuron(1));
                for(int j=1;j<topology[i]+1;j++){
                    Layers[i].push_back(Neuron(0));
                } 
            }
        }

        void initializeWeights() {
            for(int i =0;i<Layers.size();i++){
                for(int j =0;j<Layers[i].size();j++){
                    if(i!=Layers.size()-1)Layers[i][j].initializeWeights(Layers[i+1].size(),j);
                    else{
                        Layers[i][j].initializeWeights(0,j);
                    }
                }
            }
        }
        // runs through all layers, and exludes the first neuron (bias neuron)
        void forwardPass(vector<double>& input) {
            
            assert(Layers.size() != 0);

            for (int i = 1; i < Layers[0].size(); i++) {
                Layers[0][i].forwardPassFirstLayer(input, i);
            }
            

            for (int i = 1; i < Layers.size(); i++) {
                for (int j = 1; j < Layers[i].size(); j++) {
                    Layers[i][j].forwardPassSigmoid(Layers[i - 1], j);
                }
            }
        }

        void backwardPass() {
            
        }


};


int main () {
    vector<int> topology = {2, 4, 1};
    vector<double> input {1, 1};
    NeuralNet a = NeuralNet(topology);
    
    a.initializeWeights();
    cout<<"Before forward Pass"<<endl;

    for(int layer_no = 0; layer_no < a.Layers.size(); layer_no++){
        
        cout<<"Layer: "<<layer_no+1<<endl;
        cout<<"<---->"<<endl;
        
        Layer layer = a.Layers[layer_no];

        for(int neuron_no = 0; neuron_no < layer.size(); neuron_no++){
            
            cout<<"Neuron "<<neuron_no+1<<": "<<endl;
            cout<<"<---->"<<endl;
            
            Neuron neuron = layer[neuron_no];
            vector<double>neuron_weights = neuron.weights;

            for(int weight_no = 0; weight_no < neuron_weights.size(); weight_no++){
                cout<<"Weight "<<weight_no+1<<": "<<neuron_weights[weight_no]<<endl;
            }

            cout<<"Output: "<<neuron.output<<endl;

            cout<<"---------------------"<<endl;
        }

        cout<<"****************************************"<<endl;
    }


    a.forwardPass(input);
    cout<<"After forward Pass"<<endl;

    for(int layer_no = 0; layer_no < a.Layers.size(); layer_no++){
        
        cout<<"Layer: "<<layer_no+1<<endl;
        cout<<"<---->"<<endl;
        
        Layer layer = a.Layers[layer_no];

        for(int neuron_no = 0; neuron_no < layer.size(); neuron_no++){
            
            cout<<"Neuron "<<neuron_no+1<<": "<<endl;
            cout<<"<---->"<<endl;
            
            Neuron neuron = layer[neuron_no];
            vector<double>neuron_weights = neuron.weights;

            for(int weight_no = 0; weight_no < neuron_weights.size(); weight_no++){
                cout<<"Weight "<<weight_no+1<<": "<<neuron_weights[weight_no]<<endl;
            }

            cout<<"Output: "<<neuron.output<<endl;

            cout<<"---------------------"<<endl;
        }

        cout<<"****************************************"<<endl;
    }

    return 0;
}