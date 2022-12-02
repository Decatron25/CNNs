
#include "ANNS.hpp"


    
    Neuron::Neuron (double output) {
        this->output = output;
        gradient = 0;
    }

    void Neuron::initializeWeights(int weightSize, int n_index) {
        srand(time(0));
        weights.resize(weightSize);
        if (n_index == 0) {
            // double temp = (double)rand()/RAND_MAX;
            double temp =1.0;
            for(int i=0;i<weightSize;i++){
                this->weights[i] = temp;
            }
            return;
        }
            
        
        for(int i=0;i<weightSize;i++){
            // this->weights[i] = (double)rand()/RAND_MAX;
            this->weights[i] = 1.0;
        }
        
    }

    //n_index is the index of neuron in the layer
    void Neuron::forwardPassFirstLayer(vector<double>& input, int n_index) {
        if (n_index == 0) {
            output = 1;
        } else {
            output = input[n_index - 1];
        } 

    }

    double Neuron::sigmoidActivation (double x) {
        // return x;
        return 1.0 / (1.0 + exp(-1.0 * x));
    }

    void Neuron::forwardPassSigmoid(vector<Neuron>& prevLayer, int n_index) {
        double sum = 0;
        // cout<<"Layers size "<<prevLayer.size()<<endl;
        for (int i = 0; i < prevLayer.size(); i++) {
            sum += prevLayer[i].output * prevLayer[i].weights[n_index];
        }
        
        output = sigmoidActivation(sum);
    }


    double Neuron::forwardPassLastLayer(vector<Neuron>& prevLayer, int n_index) {
        double sum = 0;
        for (int i = 0; i < prevLayer.size(); i++) {
            sum += prevLayer[i].output * prevLayer[i].weights[n_index];
        }
        // cout<<"last node value "<<sum<<endl;
        output = exp(sum);
        return output;
    }

    void Neuron::backwardPass() {

    }


        NeuralNet:: NeuralNet(){

        }
        NeuralNet::NeuralNet(vector<int> topology,double eta) {
            Layers.resize(topology.size());
            for(int i=0;i<topology.size();i++){
                Layers[i].push_back(Neuron(1));
                for(int j=1;j<topology[i]+1;j++){
                    Layers[i].push_back(Neuron(0));
                } 
            }
            this->eta = eta;
        }

        void NeuralNet::initializeWeights() {
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
        void NeuralNet::forwardPass(vector<double>& input) {
            
            assert(Layers.size() != 0);

            for (int i = 1; i < Layers[0].size(); i++) {
                Layers[0][i].forwardPassFirstLayer(input, i);
            }
            

            for (int i = 1; i < Layers.size()-1; i++) {
                for (int j = 1; j < Layers[i].size(); j++) {
                    Layers[i][j].forwardPassSigmoid(Layers[i - 1], j);
                }
            }

            int lastlayer = Layers.size()-1;
            double normalize_factor = 0;
            for (int j = 1; j < Layers[lastlayer].size(); j++) {
                    normalize_factor += Layers[lastlayer][j].forwardPassLastLayer(Layers[lastlayer - 1], j);
            }
            for (int j = 1; j < Layers[lastlayer].size(); j++) {
                    Layers[lastlayer][j].output /= normalize_factor;
            }
            return;
        }


        void NeuralNet::backwardPass(vector<double>& trueOutput) {
            //compute graidents for last layer
            int lastLayer = Layers.size() - 1;
            for (int i = 1; i < Layers[lastLayer].size(); i++) {
                Layers[lastLayer][i].gradient = Layers[lastLayer][i].output - trueOutput[i - 1];
            }

            // for all prev layers compute gradients update weights
            for (int i = lastLayer - 1; i >= 0; i--) {
            
                //compute gradients
                for (int neuron = 1; neuron < Layers[i].size(); neuron++) {
                    double grad = 0;
                    for (int w = 1; w < Layers[i][neuron].weights.size(); w++) {
                        grad += Layers[i][neuron].weights[w] * Layers[i + 1][w].gradient;
                    }
                    Layers[i][neuron].gradient = grad * Layers[i][neuron].output * (1 - Layers[i][neuron].output);
                }

                //update weights
                for (int neuron = 0; neuron < Layers[i].size(); neuron++) {
                    for (int w = 1; w < Layers[i][neuron].weights.size(); w++) {
                        double delta = Layers[i + 1][w].gradient * Layers[i][neuron].output;
                        Layers[i][neuron].weights[w] -= this->eta * delta;
                    }
                }
            }

        }



//  int main () {
//     vector<int> topology = {2, 4, 2};
//     vector<double> input {1, 1};
//     vector<double> trueOuput = {1, 1};
//     double eta = 0.1;
//     NeuralNet a = NeuralNet(topology, eta);
    
//     a.initializeWeights();
//     cout<<"Before forward Pass"<<endl;

//     for(int layer_no = 0; layer_no < a.Layers.size(); layer_no++){
        
//         cout<<"Layer: "<<layer_no+1<<endl;
//         cout<<"<---->"<<endl;
        
//         vector<Neuron> layer = a.Layers[layer_no];

//         for(int neuron_no = 0; neuron_no < layer.size(); neuron_no++){
            
//             cout<<"Neuron "<<neuron_no+1<<": "<<endl;
//             cout<<"<---->"<<endl;
            
//             Neuron neuron = layer[neuron_no];
//             vector<double>neuron_weights = neuron.weights;

//             for(int weight_no = 0; weight_no < neuron_weights.size(); weight_no++){
//                 cout<<"Weight "<<weight_no+1<<": "<<neuron_weights[weight_no]<<endl;
//             }

//             cout<<"Output: "<<neuron.output<<endl;

//             cout<<"---------------------"<<endl;
//         }

//         cout<<"****************************************"<<endl;
//     }


//     a.forwardPass(input);
//     cout<<"After forward Pass"<<endl;

//     for(int layer_no = 0; layer_no < a.Layers.size(); layer_no++){
        
//         cout<<"Layer: "<<layer_no+1<<endl;
//         cout<<"<---->"<<endl;
        
//          vector<Neuron> layer = a.Layers[layer_no];

//         for(int neuron_no = 0; neuron_no < layer.size(); neuron_no++){
            
//             cout<<"Neuron "<<neuron_no+1<<": "<<endl;
//             cout<<"<---->"<<endl;
            
//             Neuron neuron = layer[neuron_no];
//             vector<double>neuron_weights = neuron.weights;

//             for(int weight_no = 0; weight_no < neuron_weights.size(); weight_no++){
//                 cout<<"Weight "<<weight_no+1<<": "<<neuron_weights[weight_no]<<endl;
//             }

//             cout<<"Output: "<<neuron.output<<endl;

//             cout<<"---------------------"<<endl;
//         }

//         cout<<"****************************************"<<endl;
//     }

//     a.backwardPass(trueOuput);
    
//     cout<<"After backward Pass"<<endl;

//     for(int layer_no = 0; layer_no < a.Layers.size(); layer_no++){
        
//         cout<<"Layer: "<<layer_no+1<<endl;
//         cout<<"<---->"<<endl;
        
//         vector<Neuron> layer = a.Layers[layer_no];

//         for(int neuron_no = 0; neuron_no < layer.size(); neuron_no++){
            
//             cout<<"Neuron "<<neuron_no+1<<": "<<endl;
//             cout<<"<---->"<<endl;
            
//             Neuron neuron = layer[neuron_no];
//             vector<double>neuron_weights = neuron.weights;

//             for(int weight_no = 0; weight_no < neuron_weights.size(); weight_no++){
//                 cout<<"Weight "<<weight_no+1<<": "<<neuron_weights[weight_no]<<endl;
//             }

//             cout<<"Gradient: "<<neuron.gradient<<endl;

//             cout<<"---------------------"<<endl;
//         }

//         cout<<"****************************************"<<endl;
//     }

//     return 0;
// }