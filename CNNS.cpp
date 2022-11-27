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
	vector<vector<vector<vector<double>>>> filterGradient;
	vector<vector<vector<double>>> layerGradient;
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
	virtual void InitializeWeights() {}
	virtual void backwardPassFirstLayer(vector<vector<vector<double>>>& outputGradients, int index, vector<Layer*>& Layers, vector<vector<vector<double>>>& inputImage) {}
	virtual void backwardPass(vector<vector<vector<double>>>& outputGradients, int index, vector<Layer*>& Layers) {}
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
	
	void un_max2D(vector<vector<double>>mat,int channel_no,int focusRow,int focusCol,double grad_val,int filterDimension){
		double maxVal = 0;
		int mat_rows = mat.size();
		int mat_cols = mat[0].size();
		int filterStartRow,filterEndRow,filterStartCol,filterEndCol;

		int maxi_row,maxi_col;

        filterStartRow = focusRow;
        filterStartCol = focusCol;
        filterEndRow = filterDimension - 1 + filterStartRow;
        filterEndCol = filterDimension - 1 + filterStartCol;

		for(int row = filterStartRow; row <= filterEndRow; row++){
			for(int col = filterStartCol; col <= filterEndCol; col++){
			    if(row<0 || row>=mat_rows || col<0 || col>=mat_cols)
				{
					if(maxVal<=0){
						maxVal = 0;
						maxi_row = row;
						maxi_col = col;
					}	
				} 
				else{
					if(maxVal<=mat[row][col]){
						maxVal = mat[row][col];
						maxi_row = row;
						maxi_col = col;
					}
					
				}
			}
		}

		this->layerGradient[channel_no][maxi_row][maxi_col]+=grad_val;

	}
	
	void backwardPass(vector<vector<vector<double>>>& outputGradients, int index, vector<Layer*>& Layers){
		vector<vector<vector<double>>>prev_output = Layers[index-1]->output;
		this->layerGradient.resize(prev_output.size(),vector<vector<double>>(prev_output[0].size(),vector<double>(prev_output[0][0].size(),0)));

		for(int channel = 0; channel < this->inputChannels; channel++){
			int row = 0;
			for(int r = startRow; r<=endRow; r+=this->stride){
				int col = 0;
				for(int c = startCol; c<=endCol; c+=this->stride){
				    un_max2D(prev_output[channel],channel,r,c,outputGradients[channel][row][col],this->filterDimension);
					col++;
				}	
				row++;
			}
		}
		
	}

	void backwardPassFirstLayer(vector<vector<vector<double>>>& outputGradients, int index, vector<Layer*>& Layers, vector<vector<vector<double>>>& inputImage) {
		
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
		this->filterGradient.resize(filterCount,vector<vector<vector<double>>>(inputChannels,vector<vector<double>>(filterDimension,vector<double>(filterDimension,0))));
		
		// cout<<"ConvLayer created"<<endl;
	}
	double relu(double sum){
		if(sum<0) sum = 0;
		return sum;
	}
	double convolution(vector<vector<vector<double>>> &mat,int row,int col,int filterDimension,vector<vector<vector<double>>> &filters){
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
					output[filter][row][col] = relu(convolution(inputImage,r,c,filterDimension,filters[filter]));
					col++;
				}	
				row++;
			}
		}
		
		
	} //apply relu

	void backwardPass(vector<vector<vector<double>>>& outputGradients, int index, vector<Layer*>& Layers){
		// resize layergradient according to previous layer output
		int inputlayerdim = Layers[index-1]->output[0].size();
		this->layerGradient.resize(filterCount,vector<vector<double>>(inputlayerdim,vector<double>(inputlayerdim,0)));
		
		//backprop for relu
		for (int i = 0; i < outputGradients.size(); i++) {
			for (int j = 0; j < outputGradients[i].size(); j++) {
				for (int k = 0; k < outputGradients[i][j].size(); k++) {
					if (output[i][j][k] == 0) {
						outputGradients[i][j][k] = 0;
					}
				}
			}
		}
		//calculation of filter gradients
		for(int filter=0;filter<filterCount;filter++){
			for(int channel=0;channel<inputChannels;channel++){
				int startind = 0;
				int endind = inputDimension-filterDimension;
				for(int i = startind ;i<=endind;i++){
					for(int j = startind;j<=endind;j++){
						for(int r=0;r<filterDimension;r++){
							for(int c = 0;c<filterDimension;c++){
								filterGradient[filter][channel][i][j] += (Layers[index-1]->output[channel][r+i][c+j]*outputGradients[filter][r][c]);
							}
						}	
					}
				}
			}
		}

		// calculation of gradients for this layer
		// flip the filter twice 
		vector<vector<vector<vector<double>>>> flipedfilters(filters);
		for(int filter = 0; filter< filterCount; filter++){
			for(int channel = 0; channel<inputChannels;channel++){
				for(int i = filterDimension-1;i>=0;i--){
					for(int j = filterDimension-1;j>=0;j-- ){
						flipedfilters[filter][channel][filterDimension-i-1][filterDimension-j-1] = filters[filter][channel][i][j];
					}
				}
			}
		}

		// full convolution b/w Filter of this layer and loss gradient of output.
		int padding_new = (inputDimension -1 + outputDimension - filterDimension)/2;
		for (int channel = 0; channel < inputChannels; channel++) {
			int startind = -1 * padding_new;
			int endind = filterDimension + padding_new - outputDimension;
			for(int i = startind ;i<=endind;i++){
					for(int j = startind;j<=endind;j++){
						for(int r=0;r<outputDimension;r++){
							for(int c = 0;c<outputDimension;c++){
								if(i+r<0 || i+r>filterDimension-1 || j+c<0 || j+c>filterDimension-1)
									continue;
								for(int filter = 0; filter<filterCount;filter++){
									layerGradient[channel][i+padding_new][j+padding_new] += outputGradients[filter][r][c]*filters[filter][channel][i+r][j+c];
								}
							}
						}	
					}
				}
		}

	}


	// backward pass for first layer
	void backwardPassFirstLayer(vector<vector<vector<double>>>& outputGradients, int index, vector<Layer*>& Layers, vector<vector<vector<double>>>& inputImage) {
		int inputlayerdim = inputImage[0].size();
		this->layerGradient.resize(filterCount,vector<vector<double>>(inputlayerdim,vector<double>(inputlayerdim,0)));
		
		//backprop for relu
		for (int i = 0; i < outputGradients.size(); i++) {
			for (int j = 0; j < outputGradients[i].size(); j++) {
				for (int k = 0; k < outputGradients[i][j].size(); k++) {
					if (output[i][j][k] == 0) {
						outputGradients[i][j][k] = 0;
					}
				}
			}
		}
		//calculation of filter gradients
		for(int filter=0;filter<filterCount;filter++){
			for(int channel=0;channel<inputChannels;channel++){
				int startind = 0;
				int endind = inputDimension-filterDimension;
				for(int i = startind ;i<=endind;i++){
					for(int j = startind;j<=endind;j++){
						for(int r=0;r<filterDimension;r++){
							for(int c = 0;c<filterDimension;c++){
								filterGradient[filter][channel][i][j] += (inputImage[channel][r+i][c+j]*outputGradients[filter][r][c]);
							}
						}	
					}
				}
			}
		}

		// calculation of gradients for this layer
		// flip the filter twice 
		vector<vector<vector<vector<double>>>> flipedfilters(filters);
		for(int filter = 0; filter< filterCount; filter++){
			for(int channel = 0; channel<inputChannels;channel++){
				for(int i = filterDimension-1;i>=0;i--){
					for(int j = filterDimension-1;j>=0;j-- ){
						flipedfilters[filter][channel][filterDimension-i-1][filterDimension-j-1] = filters[filter][channel][i][j];
					}
				}
			}
		}

		// full convolution b/w Filter of this layer and loss gradient of output.
		int padding_new = (inputDimension -1 + outputDimension - filterDimension)/2;
		for (int channel = 0; channel < inputChannels; channel++) {
			int startind = -1 * padding_new;
			int endind = filterDimension + padding_new - outputDimension;
			for(int i = startind ;i<=endind;i++){
					for(int j = startind;j<=endind;j++){
						for(int r=0;r<outputDimension;r++){
							for(int c = 0;c<outputDimension;c++){
								if(i+r<0 || i+r>filterDimension-1 || j+c<0 || j+c>filterDimension-1)
									continue;
								for(int filter = 0; filter<filterCount;filter++){
									layerGradient[channel][i+padding_new][j+padding_new] += outputGradients[filter][r][c]*flipedfilters[filter][channel][i+r][j+c];
								}
							}
						}	
					}
				}
		}
	}
	

	

	
	void InitializeWeights() {
		srand(time(0));
		
		for(int i=0;i<this->filterCount;i++){
			for(int j=0;j<this->inputChannels;j++){
				for(int k =0;k<this->filterDimension;k++){
					for(int l=0;l<this->filterDimension;l++){
						// this->filters[i][j][k][l] = (float)rand()/RAND_MAX;
						this->filters[i][j][k][l] = 1;
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
	vector<vector<vector<double>>>& inputImage;
	
	CNNnet(vector<vector<int>>& networkTopology, vector<vector<vector<double>>>& inputImageData): inputImage(inputImageData) {
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

	void backwardPass(vector<double>& gradientsOfANN) {
		
		//unflatten into unflattenLayer
		int dim1 = Layers[Layers.size() - 1]->output.size();
		int dim2 = Layers[Layers.size() - 1]->output[0].size();
		int dim3 = Layers[Layers.size() - 1]->output[0][0].size();

		vector<vector<vector<double>>> unflattenedGradients(Layers[Layers.size() - 1]->output);

		int count = 0;
		for (int i = 0; i < dim1; i++) {
			for (int j = 0; j < dim2; j++) {
				for (int k = 0; k < dim3; k++) {
					unflattenedGradients[i][j][k] = gradientsOfANN[count++];
				}
			}
		}


		//
		
		int lastlayer = topology.size() - 1;
		Layers[lastlayer]->backwardPass(unflattenedGradients, lastlayer, Layers);
		for (int i = lastlayer - 1; i > 0; i--) {
			//backwardPass()
			Layers[i]->backwardPass(Layers[i+1]->layerGradient, i, Layers);
		}

		Layers[0]->backwardPassFirstLayer(Layers[1] ->layerGradient,0, Layers, inputImage);

		
	}

	//returns the flattened version of the final pooled layer 
	
};

void print_filters(Layer* layer){
	vector<vector<vector<vector<double>>>>filters = layer->filters;

	for(int filter_no = 0; filter_no<filters.size(); filter_no++){
		cout<<"Filter no: "<<filter_no<<endl;
		cout<<"*****************"<<endl;
		for(int i=0; i< filters[0].size(); i++){
			cout<<"channel no: "<<i<<endl;
			cout<<"<---------------->"<<endl;
			for(int j=0; j<filters[0][0].size(); j++){
				for(int k=0; k<filters[0][0][0].size(); k++){
					cout<<filters[filter_no][i][j][k]<<" ";
				}
				cout<<endl;
			}
			cout<<endl;
		}
		cout<<endl;
	}
}


int main () {
	vector<vector<vector<double>>> inputImage = {{{1,2,3,4,5},{1,2,3,4,5},{1,2,3,4,5},{1,2,3,4,5},{1,2,3,4,5}}, {{1,2,3,4,5},{1,2,3,4,5},{1,2,3,4,5},{1,2,3,4,5},{1,2,3,4,5}}};
    //type of layer, number of filters, filter dimension, padding, stride
	vector<vector<int>> networkTopology = {{CONVOLUTION, 2, 3, SAME, 2}, {MAXPOOLING, 1, 3, SAME, 2}};
//    vector<vector<int>> networkTopology = {{MAXPOOLING, 1, 3, SAME, 1}};

	CNNnet CNN(networkTopology, inputImage);
	CNN.InitializeLayers();
	CNN.forwardPass(inputImage);
	
	
	for(int layer_no = 0;layer_no<CNN.Layers.size();layer_no++){
		
		cout<<"Layer_no: "<<layer_no<<endl;
		cout<<"$$$$$$$$$$$$$$$$$$$$$$$$$"<<endl;

		cout<<"Filters: "<<endl;
	
		print_filters(CNN.Layers[layer_no]);
		vector<vector<vector<double>>>& layer_output = CNN.Layers[layer_no]->output;
		
		cout<<"No.of channels: "<<layer_output.size()<<endl;
		cout<<"output dimension: "<<layer_output[0].size()<<endl;

		cout<<"Output: "<<endl;
		cout<<"<-------->"<<endl;
		for(int channel_no = 0; channel_no<layer_output.size();channel_no++){
			cout<<"channel_no: "<<channel_no<<endl;
			for(int row = 0; row<layer_output[0].size(); row++){
				
				for(int col = 0; col<layer_output[0][0].size(); col++){
					cout<<layer_output[channel_no][row][col]<<" ";
				}
				cout<<endl;
			}
			cout<<endl;
		}
		cout<<"<--------------------->"<<endl;
	}
	vector<double> v(50, 0.1);
	CNN.backwardPass(v);

	cout<<"BACKWARD PASS COMPLETED"<<endl;
	cout<<"<--------------->"<<endl;
	cout<<endl;
	cout<<endl;

	for(int layer_no = 0;layer_no<CNN.Layers.size();layer_no++){
		
		cout<<"Layer_no: "<<layer_no<<endl;
		cout<<"$$$$$$$$$$$$$$$$$$$$$$$$$"<<endl;

		cout<<"Filters: "<<endl;
	
		print_filters(CNN.Layers[layer_no]);
		vector<vector<vector<double>>>& layer_output = CNN.Layers[layer_no]->output;
		
		cout<<"No.of channels: "<<layer_output.size()<<endl;
		cout<<"output dimension: "<<layer_output[0].size()<<endl;

		cout<<"Output: "<<endl;
		cout<<"<-------->"<<endl;
		for(int channel_no = 0; channel_no<layer_output.size();channel_no++){
			cout<<"channel_no: "<<channel_no<<endl;
			for(int row = 0; row<layer_output[0].size(); row++){
				
				for(int col = 0; col<layer_output[0][0].size(); col++){
					cout<<layer_output[channel_no][row][col]<<" ";
				}
				cout<<endl;
			}
			cout<<endl;
		}
		cout<<"<--------------------->"<<endl;
	}
	

	return 0;
}