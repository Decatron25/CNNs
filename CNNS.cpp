#include<bits/stdc++.h>
#include<random>
#include<time.h>
#include <cstdlib>
using namespace std;

enum {
	CONVOLUTION,
	MAXPOOLING,
	AVGPOOLING,
    VALID,
    SAME
};


class Layer {

public:
	vector<vector<vector<vector<double>>>> filters;
	vector<double> biases;
	vector<vector<vector<int>>> output;
	int inputChannels;
	int inputDimension;
	int outputDimension;
	int filterCount;
	int filterDimension;
	int padding;
	int stride;

	virtual void forwardPass(vector<vector<vector<int>>> inputImage) {} //apply relu
	virtual void InitializeWeights() {}
};

class PoolingLayer: public Layer{

public: 
	int startRow,startCol,endRow,endCol;
	PoolingLayer (int inputChannels, int filterCount, int filterDimension, int padding = 0, int stride = 1, string type = "Max") {
		this->inputChannels = inputChannels;
		this->filterCount = 1;
		this->padding = padding;
		this->stride = stride;
		this->filterDimension = filterDimension;

		if(this->padding!=0){
			this->padding = (this->stride-1)*(this->inputDimension) - this->stride + this->filterDimension ;
		}
        
    }
	int max_finder2D(vector<vector<int>>mat,int focusRow,int focusCol,int filterDimension){
		int maxVal = mat[focusRow][focusCol];
		int filterHalf = (filterDimension-1)/2;
		int filterStartrow,filterEndrow,filterStartcol,filterEndcol;
		
		filterStartrow = focusRow;
		filterStartcol = focusCol;
		filterEndrow = filterDimension-1;
		filterEndcol = focusCol+;

		for(int row = filterStartrow; row<=filterEndrow; row++){
			for(int col = filterEndrow; col<=filterEndcol; col++){
				maxVal = max(maxVal,mat[row][col]);
			}
		}

		return maxVal;
	}
	
	void forwardPass(vector<vector<vector<int>>> inputImage) {
		assert(this->inputChannels == inputImage.size());
		this->inputDimension = inputImage[0].size();
		this->outputDimension = floor((this->inputDimension-this->filterDimension+(2*(this->padding)))/this->stride)+1;

		startRow = -1*(this->padding);
		startCol = -1*(this->padding);
		endRow = this->inputDimension+this->padding-1-filterDimension;
		endCol = endRow;

		output.resize(this->inputChannels,vector<vector<int>>(this->outputDimension,vector<int>(this->outputDimension)));
		
		for(int channel = 0; channel < this->inputChannels; channel++){
			int row = 0;
			for(int r = startRow; r<endRow; r+=this->stride){
				int col = 0;
				for(int c = startCol; c<endCol; c+=this->stride){
					output[row][col][channel] = max_finder2D(inputImage[channel],r,c,this->filterDimension);
					col++;
				}	
				row++;
			}
		}
	} 

	void InitializeWeights(){
		return;
	}
};

class ConvLayer: public Layer{

public:
	int startRow,startCol,endRow,endCol;
	ConvLayer (int inputChannels, int filterCount, int filterDimension, int padding = 0, int stride = 1) {
		this->inputChannels = inputChannels;
		this->filterCount = filterCount;
		this->stride = stride;
		this->padding = padding;
		this->filters.resize(filterCount,vector<vector<vector<double>>>(inputChannels,vector<vector<double>>(filterDimension,vector<double>(filterDimension))));
		cout<<"ConvLayer created";
	}

	int convolution(vector<vector<int>>mat,int focusRow,int focusCol,int filterDimension,int padding){
		
	}
	void forwardPass(vector<vector<vector<int>>> inputImage) {
		assert(this->inputChannels == inputImage.size());
		this->inputDimension = inputImage[0].size();
		
		if(this->padding!=0){
			this->padding = (this->stride-1)*(this->inputDimension) - this->stride + this->filterDimension ;
		}
		
		this->outputDimension = this->outputDimension = floor((this->inputDimension-this->filterDimension+(2*(this->padding)))/this->stride)+1;
		
		startRow = 0;
		startCol = 0;
		endRow = this->inputDimension-1-filterDimension;
		endCol = endRow;

		output.resize(this->inputChannels,vector<vector<int>>(this->outputDimension,vector<int>(this->outputDimension)));
		
		for(int channel = 0;channel<this->inputChannels;channel++){
			int row = 0;
			for(int r = startRow ; r<endRow; r+=this->stride){
				int col = 0;
				for(int c = startCol;c<endCol;c+=this->stride){
					output[row][col][channel] = convolution(inputImage[channel],r,c,this->filterDimension,this->padding);
					col++;
				}	
				row++;
			}
		}
		
	} //apply relu

	void InitializeWeights() {
		srand(time(0));
		
		for(int i=0;i<this->filterCount;i++){
			for(int j=0;j<this->inputChannels;j++){
				for(int k =0;k<this->filterDimension;k++){
					for(int l=0;l<this->filterDimension;l++){
						this->filters[i][j][k][l] = rand()%RAND_MAX;
					}
				}
			}
		}
		
	}

};


class CNNnet 
{

public:
	vector<Layer*> Layers;
	vector<int> topology;
	

	CNNnet(vector<vector<int>>& networkTopology, vector<vector<vector<int>>>& inputImage) {
        //assumption: first layer is a CONVOLUTION layer
        if (networkTopology.size() == 0) {
            cout<<"give valid topology"<<endl;
        } else if (networkTopology[0][0] != CONVOLUTION) {
            cout<<"first layer should be a convolution layer"<<endl;
        } else {
            int inputChannels = inputImage.size();
            int filterCount = networkTopology[0][1];
            int filterDimension = networkTopology[0][2];
            int padding = networkTopology[0][3];
            int stride = networkTopology[0][4];
            
            Layer* newLayer = new ConvLayer(inputChannels, filterCount, filterDimension, padding, stride);
            
            
            //Layers.push_back(new ConvLayer(inputImage.size(), networkTopology[0][1], networkTopology[0][2], networkTopology[0][3], networkTopology[0][4]));
            Layers.push_back(newLayer);
            topology.push_back(networkTopology[0][0]);
            for (int i = 1; i < networkTopology.size(); i++) {
                inputChannels = Layers[i - 1]->filterCount;
                filterCount = networkTopology[i][1];
                filterDimension = networkTopology[i][2];
                padding = networkTopology[i][3];
                stride = networkTopology[i][4];

                if (networkTopology[i][0] == CONVOLUTION) {
                    newLayer = new ConvLayer(inputChannels, filterCount, filterDimension, padding, stride);
                    Layers.push_back(newLayer);
                } else if (networkTopology[i][0] == MAXPOOLING) {
                    newLayer = new PoolingLayer(inputChannels, filterCount, filterDimension, padding, stride, "Max");
                    Layers.push_back(newLayer);
                } else {
                    newLayer = new PoolingLayer(inputChannels, filterCount, filterDimension, padding, stride, "Avg");
                    Layers.push_back(newLayer);
                }

                topology.push_back(networkTopology[i][0]);
            }    
        }
        
	}

	void forwardPass(vector<vector<vector<int>>>& inputImage) {
        Layers[0]->forwardPass(inputImage);
		for (int i = 1; i < topology.size(); i++) {
            Layers[i]->forwardPass(Layers[i - 1]->output);
        }
	}

	void InitializeLayers() {
        for (int i = 0; i < topology.size(); i++) {
            Layers[i]->InitializeWeights();
        }
	}

	
};



int main () {
	vector<vector<vector<int>>> inputImage = {{}};
    //type of layer, number of filters, filter dimension, padding, stride
	vector<vector<int>> networkTopology = {{CONVOLUTION, 3, 5, 0, 1}, {MAXPOOLING, 3, 5, , 1}};


	CNNnet CNN(networkTopology, inputImage);
	CNN.forwardPass(inputImage);

    return 0;
}