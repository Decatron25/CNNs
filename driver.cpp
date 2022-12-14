#include "CNNS.hpp"
#include "ANNS.hpp"


class FullNN {
public:
    CNNnet CNNnetObj;
    NeuralNet ANNObj;
    vector<vector<vector<vector<double>>>> inputData;
    vector<vector<double>> trueOutputData;

    void getData(string filename);

    void printData () {
        for (int i = 0; i < inputData.size(); i++) {
            cout<<endl;
            cout<< "Image no: "<<i<<endl;
            cout<<"Image dimensions: "<<inputData[i][0].size()<<endl;
            for (int j = 0; j < inputData[i].size(); j++) {
                for (int k = 0; k < inputData[i][j].size(); k++) {
                    for (int l = 0; l < inputData[i][j][k].size(); l++) {
                        cout<<inputData[i][j][k][l]<<" ";
                    }
                }
            }
        }
    }

    FullNN (string filename) {
        getData(filename);
        // printData();
        vector<int> topology = {800, 128, 10};
        double eta = 0.1;
        ANNObj = NeuralNet(topology, eta);
        ANNObj.initializeWeights();

        vector<vector<int>> networkTopology = {{CONVOLUTION, 8, 5, VALID, 1},{CONVOLUTION, 8, 5, VALID, 1}, {MAXPOOLING, 1, 2, VALID, 2}};
        double learning_rate  = 0.15;
    	CNNnetObj = CNNnet(networkTopology, inputData[0], learning_rate);
        CNNnetObj.InitializeLayers();
    

    }

    void forwardPass(vector<vector<vector<double>>>& inputData) {
        CNNnetObj.forwardPass(inputData);
        cout<<CNNnetObj.flattenedOutput[799]<<endl;
        ANNObj.forwardPass(CNNnetObj.flattenedOutput);
        cout<<"driver forward pass done"<<endl;
    }

    void backwardPass(vector<double>& trueOuput) {
        ANNObj.backwardPass(trueOuput);
        vector<double> gradientsToCNN;
        for (int i = 0; i < ANNObj.Layers[0].size(); i++) {
            gradientsToCNN.push_back(ANNObj.Layers[0][i].gradient);
        }
        CNNnetObj.backwardPass(gradientsToCNN);
        cout<<"Driver backward pass done"<<endl;
    }


    void training (int iterations) {
        cout<<"entered training"<<endl;
        for (int i = 0; i < iterations; i++) {
            cout<<"Inputdata size: "<< inputData.size()<<endl;
            for (int j  = 0; j < 2; j++) {
                cout<<j<<endl;
                this->forwardPass(inputData[j]);
                this->backwardPass(trueOutputData[j]);
            }
        }
        cout<<"training done "<<endl;
    }

    void test (string filename) {
        //perform forward pass, check predicted label vs actual label
        //calculate error of the model
    }


};

void FullNN::getData(string filename) {
        vector<double> row;
		string line, word;
		int rowCount = 0;
		fstream file (filename, ios::in);
		if(file.is_open())
		{
			while(getline(file, line))
			{


				rowCount++;
				if (rowCount == 1) {
					continue;
				}
				row.clear();
				 
				stringstream str(line);



                int lastColVal;
				while(getline(str, word, ',')) {
                    lastColVal = stoi(word);
                    row.push_back(stod(word));
                }

                vector<double> trueOutput(10, 0);
                trueOutput[lastColVal] = 1.0;
                trueOutputData.push_back(trueOutput);

				if (row.size() > 0)
					row.pop_back();
				//cout<<rowCount<<" "<<row[129]<<endl;
                int dim = 28;
                vector<vector<vector<double>>> inputImage (1, vector<vector<double>> (dim, vector<double> (dim, 0)));
                for (int i = 0; i < row.size(); i++) {
                    int r = i/dim;
                    int c = i%dim;
                    inputImage[0][r][c] = row[i]/256;
                }
                

				inputData.push_back(inputImage);
	
			}
		}
};

int main () {
    int iterations = 1;
    FullNN n = FullNN("train.csv");
    n.training(iterations);
    // n.test("test.csv");

    cout<<"After forward Pass"<<endl;

    for(int layer_no = 0; layer_no < n.ANNObj.Layers.size(); layer_no++){
        
        cout<<"Layer: "<<layer_no+1<<endl;
        cout<<"<---->"<<endl;
        
         vector<Neuron> layer = n.ANNObj.Layers[layer_no];

        for(int neuron_no = layer.size()-10; neuron_no < layer.size(); neuron_no++){
            // if(neuron_no>layer.size()-10)
            cout<<"Neuron "<<neuron_no+1<<": "<<endl;
            cout<<"<---->"<<endl;
            
            Neuron neuron = layer[neuron_no];
            // vector<double>neuron_weights = neuron.weights;

            // for(int weight_no = 0; weight_no < neuron_weights.size(); weight_no++){
            //     cout<<"Weight "<<weight_no+1<<": "<<neuron_weights[weight_no]<<endl;
            // }
            
            cout<<"Output: "<<neuron.output<<endl;

            cout<<"---------------------"<<endl;
        }

        cout<<"****************************************"<<endl;
    }

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

}