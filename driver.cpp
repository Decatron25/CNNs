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
        printData();
        vector<int> topology = {2, 4, 2};
        double eta = 0.1;
        ANNObj = NeuralNet(topology, eta);
        ANNObj.initializeWeights();

        vector<vector<int>> networkTopology = {{CONVOLUTION, 3, 2, VALID, 1}, {MAXPOOLING, 1, 2, VALID, 2}};
        double learning_rate  = 0.15;
    	CNNnetObj = CNNnet(networkTopology, inputData[0], learning_rate);
        CNNnetObj.InitializeLayers();
    

    }

    void forwardPass(vector<vector<vector<double>>>& inputData) {
        CNNnetObj.forwardPass(inputData);
        ANNObj.forwardPass(CNNnetObj.flattenedOutput);
    }

    void backwardPass(vector<double>& trueOuput) {
        ANNObj.backwardPass(trueOuput);
        vector<double> gradientsToCNN;
        for (int i = 0; i < ANNObj.Layers[0].size(); i++) {
            gradientsToCNN.push_back(ANNObj.Layers[0][i].gradient);
        }
        CNNnetObj.backwardPass(gradientsToCNN);
    }


    void training (int iterations) {

        for (int i = 0; i < iterations; i++) {
            for (int j  = 0; j < inputData.size(); j++) {
                this->forwardPass(inputData[j]);
                this->backwardPass(trueOutputData[j]);
            }
        }
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
                    inputImage[0][r][c] = row[i];
                }
                

				inputData.push_back(inputImage);
	
			}
		}
};

int main () {
    int iterations = 1;
    FullNN n = FullNN("train.csv");
    n.training(iterations);
    n.test("test.csv");


}