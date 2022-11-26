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
	vector<vector<vector<double>>> output;
	int inputChannels;
	int inputDimension;
	int outputDimension;
	int filterCount;
	int filterDimension;
	int padding;
	int stride;

	virtual void forwardPass(vector<vector<vector<double>>> &inputImage) {} //apply relu
	virtual void  InitializeWeights() {}
};

class PoolingLayer: public Layer{

public:
	int startRow,startCol,endRow,endCol;
	string poolingType;

	PoolingLayer (int inputChannels, int filterCount, int filterDimension, int padding = VALID, int stride = 1, string type = "Max") {
		this->inputChannels = inputChannels;
		this->filterCount = 1;
		this->padding = padding;
		this->stride = stride;
		this->filterDimension = filterDimension;
		this->poolingType = type;


        // cout<<"Pooling Layer created"<<endl;
    }
	double max_finder2D(vector<vector<double>>mat,int focusRow,int focusCol,int filterDimension){
		double maxVal = 0;
		int mat_rows = mat.size();
		int mat_cols = mat[0].size();
		int filterStartRow,filterEndRow,filterStartCol,filterEndCol;

        filterStartRow = focusRow;
        filterStartCol = focusCol;
        filterEndRow = filterDimension - 1 + filterStartRow;
        filterEndCol = filterDimension - 1 + filterStartCol;

		for(int row = filterStartRow; row <= filterEndRow; row++){
			for(int col = filterStartCol; col <= filterEndCol; col++){
			    if(row<0 || row>=mat_rows || col<0 || col>=mat_cols) maxVal = max((double)maxVal,(double)0);
				else maxVal = max((double)maxVal,(double)mat[row][col]);
			}
		}

		return maxVal;
	}

    double avg_finder2D(vector<vector<double>>mat,int focusRow,int focusCol,int filterDimension){
        double avgVal = 0;
        int mat_rows = mat.size();
        int mat_cols = mat[0].size();
        int filterStartRow,filterEndRow,filterStartCol,filterEndCol;

        filterStartRow = focusRow;
        filterStartCol = focusCol;
        filterEndRow = filterDimension - 1 + filterStartRow;
        filterEndCol = filterDimension - 1 + filterStartCol;


        for(int row = filterStartRow; row <= filterEndRow; row++){
            for(int col = filterStartCol; col <= filterEndCol; col++){
                if(row<0 || row>=mat_rows || col<0 || col>=mat_cols) avgVal+=0;
                else avgVal+=(mat[row][col]);
            }
        }

        avgVal = avgVal/(filterDimension*filterDimension);

        return avgVal;
    }

	void forwardPass(vector<vector<vector<double>>> &inputImage) {
		assert(this->inputChannels == inputImage.size());
		// cout<<"1"<<endl;
		this->inputDimension = inputImage[0].size();

		if(this->padding==SAME){
            this->padding = ((this->stride-1)*(this->inputDimension) - this->stride + this->filterDimension)/2;
			// cout<<"padding in constructor:"<<this->padding<<endl;
		}
		else if(this->padding==VALID){
		    this->padding = 0;
		}
        else{//should actually throw an error or exception
            cout<<"Padding can be either SAME or VALID, no other values permissible";
        }

		this->outputDimension = floor((this->inputDimension-this->filterDimension+(2*(this->padding)))/this->stride)+1;

		// cout<<"padding: "<<this->padding<<endl;
		startRow = -1*(this->padding);
		startCol = startRow;
		endRow = this->inputDimension+this->padding- this->filterDimension;
		endCol = endRow;


		output.resize(this->inputChannels,vector<vector<double>>(this->outputDimension,vector<double>(this->outputDimension)));
		// cout<< inputChannels<<" "<<outputDimension<<endl;
		// cout<<"2"<<endl;
		
		for(int channel = 0; channel < this->inputChannels; channel++){
		    // cout<<"channel: "<<channel<<endl;
			int row = 0;
			for(int r = startRow; r<=endRow; r+=this->stride){
                // cout<<"row: "<<row<<endl;
                // cout<<"r: "<<r<<endl;
				int col = 0;
				for(int c = startCol; c<=endCol; c+=this->stride){
                    // cout<<"col: "<<col<<endl;
                    // cout<<"c: "<<c<<endl;
				    if(this->poolingType == "Max") {
                        // cout<<"3.0"<<endl;
				        output[channel][row][col] = max_finder2D(inputImage[channel],r,c,this->filterDimension);
				        // cout<<"3"<<endl;
				    }
				    else {
				        output[channel][row][col] = avg_finder2D(inputImage[channel],r,c,this->filterDimension);
				        // cout<<"4"<<endl;
				    }
				    // cout<<"5"<<endl;
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
	ConvLayer (int inputChannels, int filterCount, int filterDimension, int padding = VALID, int stride = 1) {
		this->inputChannels = inputChannels;
		this->filterCount = filterCount;
		this->filterDimension = filterDimension;
		this->stride = stride;
		this->padding = padding;
		this->filters.resize(filterCount,vector<vector<vector<double>>>(inputChannels,vector<vector<double>>(filterDimension,vector<double>(filterDimension))));
		// cout<<"ConvLayer created"<<endl;
	}

	double convolution_relu(vector<vector<vector<double>>> &mat,int row,int col,int filterDimension,vector<vector<vector<double>>> &filters){
		double sum = 0;
		int inputchannels = mat.size();
		int imageDim = mat[0].size();
		for(int i=0;i<filterDimension;i++){
			for(int j=0;j<filterDimension;j++){
				for(int channel = 0;channel<inputChannels;channel++){
					sum += ((row+i<0 || col+j<0 || row+i>=imageDim || col+j >= imageDim ) ? 0 : (filters[channel][i][j]*mat[channel][row+i][col+j]+biases[channel]));
				}
			}
		}
		if(sum<0) sum = 0;
		return sum;
	}

	void forwardPass(vector<vector<vector<double>>> &inputImage) {
		assert(inputChannels == inputImage.size());
		this->inputDimension = inputImage[0].size();
		
		if(this->padding!= VALID){
			this->padding = ((this->stride-1)*(this->inputDimension) - this->stride + this->filterDimension)/2 ;
		}
		else{
			this->padding = 0;
		}
		// assert(this->padding < this->filterDimension);
		
		this->outputDimension = floor((this->inputDimension-this->filterDimension+(2*(this->padding)))/this->stride)+1;
		
		startRow = -1*(this->padding);
		startCol = -1*(this->padding);
		endRow = this->inputDimension- this->filterDimension + this->padding;
		endCol = endRow;
		output.resize(this->filterCount,vector<vector<double>>(this->outputDimension,vector<double>(this->outputDimension)));
		for(int filter = 0;filter<this->filterCount;filter++){
			int row = 0;
			for(int r = startRow ; r<=endRow; r+=stride){
				int col = 0;
				for(int c = startCol;c<=endCol;c+=stride){
					output[filter][row][col] = convolution_relu(inputImage,r,c,filterDimension,filters[filter]);
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
						this->filters[i][j][k][l] = (float)rand()/RAND_MAX;
						// cout<< filters[i][j][k][l] << " ";
					}
					// cout<<endl;
				}
			}
		}
		// cout<<"Initialised weights"<<endl;
		this->biases.resize(filterCount,(double)0);
	}

};


class CNNnet 
{

public:
	vector<Layer*> Layers;
	vector<int> topology;
	

	CNNnet(vector<vector<int>>& networkTopology, vector<vector<vector<double>>>& inputImage) {
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
				// cout<<"inputchannles:"<<inputChannels<<endl;
                filterCount = networkTopology[i][1];
                filterDimension = networkTopology[i][2];
                padding = networkTopology[i][3];
				// cout<<"padding:"<<padding<<endl;
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

	void forwardPass(vector<vector<vector<double>>>& inputImage) {
        Layers[0]->forwardPass(inputImage);
        // cout<<"1st forward"<<endl;
		for (int i = 1; i < topology.size(); i++) {
           Layers[i]->forwardPass(Layers[i - 1]->output);
        //    cout<<"forward loop"<<endl;
       }
	}

	void InitializeLayers() {
        for (int i = 0; i < topology.size(); i++) {
            Layers[i]->InitializeWeights();
        }
	}

	
};



int main () {
	vector<vector<vector<double>>> inputImage = {{{1,2,3,4,5},{6,7,8,9,10},{11,12,13,14,15},{16,17,18,19,20},{21,22,23,24,25}}};
    //type of layer, number of filters, filter dimension, padding, stride
	vector<vector<int>> networkTopology = {{CONVOLUTION, 1, 3, SAME, 2}, {AVGPOOLING, 1, 3, SAME, 2}};
//    vector<vector<int>> networkTopology = {{MAXPOOLING, 1, 3, SAME, 1}};

	CNNnet CNN(networkTopology, inputImage);
	CNN.InitializeLayers();
	CNN.forwardPass(inputImage);
	// cout<<"after CNN forward"<<endl;

	vector<vector<vector<double>>>layer0_output = CNN.Layers[0]->output;
    vector<vector<vector<double>>>layer1_output = CNN.Layers[1]->output;

	int layer0_channels = layer0_output.size();
	int layer0_dim = layer0_output[0].size();
	// cout<< layer0_output.size()<<" "<<layer0_output[0].size()<<endl;
	for(int i=0;i<layer0_channels;i++){
	    for(int j=0;j<layer0_dim;j++){
	        for(int k=0;k<layer0_dim;k++){
	            cout<<layer0_output[i][j][k]<<" ";
	        }
	        cout<<endl;
	    }
	    cout<<endl;
	}

	// cout<<"layer 1"<<endl;
   int layer1_channels = layer1_output.size();
//    cout<<layer1_channels<<endl;
   int layer1_dim = layer1_output[0].size();
   cout<<layer1_channels<<" "<<layer1_dim<<endl;
   for(int i=0;i<layer1_channels;i++){
       for(int j=0;j<layer1_dim;j++){
           for(int k=0;k<layer1_dim;k++){
               cout<<layer1_output[i][j][k]<<" ";
           }
           cout<<endl;
       }
       cout<<endl;
   }

	return 0;
}