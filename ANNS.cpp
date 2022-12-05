
#include "ANNS.hpp"


    
    Neuron::Neuron (double output) {
        this->output = output;
        gradient = 0;
    }

    void Neuron::initializeWeights(int weightSize, int n_index,default_random_engine &generator,normal_distribution<double> &distribution) {
        // srand(time(0));
        weights.resize(weightSize);

        // default_random_engine generator(time(0));
        // cout<<time(0);
        // normal_distribution<double> distribution(0.0,0.5);

        if (n_index == 0) {
            // double temp = (double)rand()/RAND_MAX;
            double temp =0;
            for(int i=0;i<weightSize;i++){
                this->weights[i] = temp;
            }
            return;
        }
            
        
        for(int i=0;i<weightSize;i++){
            double number = distribution(generator);
            // cout<<"random number generator "<<number<<endl;
            this->weights[i] = number*0.1;
            // if(i==weightSize-1) cout<<"Weights in ANNS " << this->weights[i]<<endl;
            // this->weights[i] = 1.0;
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
         return max(x, 0.0);
    }

    void Neuron::forwardPassSigmoid(vector<Neuron>& prevLayer, int n_index) {
        double sum = 0;
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
        output = exp(sum);
        return output;
    }

    void Neuron::backwardPass() {

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

        default_random_engine generator(time(0));
        normal_distribution<double> distribution(0.0,1);

        for(int i =0;i<Layers.size();i++){
            for(int j =0;j<Layers[i].size();j++){
                if(i!=Layers.size()-1) 
                    Layers[i][j].initializeWeights(Layers[i+1].size(),j, generator, distribution);
                else
                    Layers[i][j].initializeWeights(0,j,generator,distribution);
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
                    if (Layers[i][neuron].output == 0)
                        Layers[i][neuron].gradient = 0;
                    else
                        Layers[i][neuron].gradient = grad;
                        
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

        void NeuralNet::train(int epochs, vector<vector<double>>& inputData, vector<vector<double>>& trueOutput) {
            for (int i = 0; i < epochs; i++) {
                cout<<"epoch "<<i<<":\n";
                for (int j = 0; j < inputData.size(); j++) {
                    forwardPass(inputData[j]);
                    backwardPass(trueOutput[j]);
                }
            }
        }


vector<double> lineToVec(string &line)
    {
        vector<double> values;
        string tmp = "";

        for (int i = 0; i < (int)line.length(); i++)
        {
            if ((48 <= int(line[i]) && int(line[i])  <= 57) || line[i] == '.' || line[i] == '+' || line[i] == '-' || line[i] == 'e')
            {
                tmp += line[i];
            }
            else if (tmp.length() > 0)
            {

                values.push_back(stod(tmp));
                tmp = "";
            }
        }
        if (tmp.length() > 0)
        {
            values.push_back(stod(tmp));
            tmp = "";
        }

        return values;
    }

void testing (NeuralNet& ANNObj) {
        //empty the 2 data structures, read input in them from test file
        vector<vector<double>> inputData;
        vector<vector<double>> trueOutputData;

        double crossEntropyLoss = 0;
        int accuratePredictions = 0;
        
        string filename = "test.txt";
        ifstream infile(filename.c_str());

        if (!infile.is_open())
        {
            cout << "Error: Failed to open file." << endl;
            return;
        }

        // Fetching points from file
    
        string line;

        cout<<"In testing"<<endl;
        while (getline(infile, line))
        {
            vector<double> row = lineToVec(line);
            int label = (int) row[row.size() - 1];
            vector<double> output (7, 0);
            output[label] = 1;
            trueOutputData.push_back(output);
            row.pop_back();
            inputData.push_back(row);
        }

        for (int i = 0; i < inputData.size(); i++) {
            ANNObj.forwardPass(inputData[i]);
            cout<<"Forward pass "<<i<<endl;
            int lastLayerIndex = ANNObj.Layers.size() - 1;
            vector<Neuron>& lastLayer = ANNObj.Layers[lastLayerIndex];
            int predictedOutput = -1;
            double maxProb = 0;
            for (int n = 1; n < lastLayer.size(); n++) {
                cout<<"Probablity for "<<n-1<<":"<<lastLayer[n].output<<endl;
                if (lastLayer[n].output > maxProb) {
                    maxProb = lastLayer[n].output;
                    predictedOutput = n-1;
                }
            }
            int trueOutput = -1;
            for (int j = 0; j < trueOutputData[i].size(); j++) {
                if (trueOutputData[i][j] != 0) {
                    trueOutput = j;
                }
            }
            cout<<"True output is: "<<trueOutput;
            cout<<", Predicted output is: "<<predictedOutput<<endl;
            if (trueOutput == predictedOutput) {
                accuratePredictions++;
            }

            crossEntropyLoss += -1 * log(lastLayer[trueOutput + 1].output);

            

            cout<<"-------------->\n";

        }

        //keep track of missclassified results, and print the accuracy finally
        cout<<"\n\nACCURACY: "<<(1.0*accuratePredictions)/inputData.size();
        cout<<"\nCrossEntropyLoss: "<<crossEntropyLoss/inputData.size();
    }

void getData(string filename, vector<vector<double>>& inputData, vector<vector<double>>& trueOutputData) {
        ifstream infile(filename.c_str());

        if (!infile.is_open())
        {
            cout << "Error: Failed to open file." << endl;
            return;
        }

        // Fetching points from file
    
        string line;

    
        while (getline(infile, line))
        {
            vector<double> row = lineToVec(line);
            int label = (int) row[row.size() - 1];
            vector<double> output (7, 0);
            output[label] = 1;
            trueOutputData.push_back(output);
            row.pop_back();
            inputData.push_back(row);
        }
}


 int main () {
    vector<int> topology = {8, 17, 12, 3, 7};
    vector<vector<double>> input;
    vector<vector<double>> trueOutput;
    string filename = "beansData.txt";
    getData(filename, input, trueOutput);
    cout<<input.size()<<endl;
    double eta = 0.005;
    NeuralNet a = NeuralNet(topology, eta);
    
    a.initializeWeights();

    a.train(7, input, trueOutput);
    testing(a);
    // cout<<"Before forward Pass"<<endl;

    // for(int layer_no = 0; layer_no < a.Layers.size(); layer_no++){
        
    //     cout<<"Layer: "<<layer_no+1<<endl;
    //     cout<<"<---->"<<endl;
        
    //     vector<Neuron> layer = a.Layers[layer_no];

    //     for(int neuron_no = 0; neuron_no < layer.size(); neuron_no++){
            
    //         cout<<"Neuron "<<neuron_no+1<<": "<<endl;
    //         cout<<"<---->"<<endl;
            
    //         Neuron neuron = layer[neuron_no];
    //         vector<double>neuron_weights = neuron.weights;

    //         for(int weight_no = 0; weight_no < neuron_weights.size(); weight_no++){
    //             cout<<"Weight "<<weight_no+1<<": "<<neuron_weights[weight_no]<<endl;
    //         }

    //         cout<<"Output: "<<neuron.output<<endl;

    //         cout<<"---------------------"<<endl;
    //     }

    //     cout<<"****************************************"<<endl;
    // }


    // a.forwardPass(input);
    // cout<<"After forward Pass"<<endl;

    // for(int layer_no = 0; layer_no < a.Layers.size(); layer_no++){
        
    //     cout<<"Layer: "<<layer_no+1<<endl;
    //     cout<<"<---->"<<endl;
        
    //      vector<Neuron> layer = a.Layers[layer_no];

    //     for(int neuron_no = 0; neuron_no < layer.size(); neuron_no++){
            
    //         cout<<"Neuron "<<neuron_no+1<<": "<<endl;
    //         cout<<"<---->"<<endl;
            
    //         Neuron neuron = layer[neuron_no];
    //         vector<double>neuron_weights = neuron.weights;

    //         for(int weight_no = 0; weight_no < neuron_weights.size(); weight_no++){
    //             cout<<"Weight "<<weight_no+1<<": "<<neuron_weights[weight_no]<<endl;
    //         }

    //         cout<<"Output: "<<neuron.output<<endl;

    //         cout<<"---------------------"<<endl;
    //     }

    //     cout<<"****************************************"<<endl;
    // }

    // a.backwardPass(trueOuput);
    
    // cout<<"After backward Pass"<<endl;

    // for(int layer_no = 0; layer_no < a.Layers.size(); layer_no++){
        
    //     cout<<"Layer: "<<layer_no+1<<endl;
    //     cout<<"<---->"<<endl;
        
    //     vector<Neuron> layer = a.Layers[layer_no];

    //     for(int neuron_no = 0; neuron_no < layer.size(); neuron_no++){
            
    //         cout<<"Neuron "<<neuron_no+1<<": "<<endl;
    //         cout<<"<---->"<<endl;
            
    //         Neuron neuron = layer[neuron_no];
    //         vector<double>neuron_weights = neuron.weights;

    //         for(int weight_no = 0; weight_no < neuron_weights.size(); weight_no++){
    //             cout<<"Weight "<<weight_no+1<<": "<<neuron_weights[weight_no]<<endl;
    //         }

    //         cout<<"Gradient: "<<neuron.gradient<<endl;

    //         cout<<"---------------------"<<endl;
    //     }

    //     cout<<"****************************************"<<endl;
    // }

    return 0;
}