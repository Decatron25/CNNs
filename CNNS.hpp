#ifndef CNNS_HPP
#define CNNS_HPP

#include <iostream>
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
	vector<double>biasesGradient;
	vector<vector<vector<double>>> output;
	int inputChannels;
	int inputDimension;
	int outputDimension;
	int filterCount;
	int filterDimension;
	int padding;
	int stride;
    double learning_rate;

	virtual void forwardPass(vector<vector<vector<double>>> &inputImage);//apply relu
	virtual void InitializeWeights();
	virtual void backwardPassFirstLayer(vector<vector<vector<double>>>& outputGradients, int index, vector<Layer*>& Layers, vector<vector<vector<double>>>& inputImage);
	virtual void backwardPass(vector<vector<vector<double>>>& outputGradients, int index, vector<Layer*>& Layers);
	virtual void updateWeights();
	virtual void updateBiases();
	virtual void backwardPassBias(vector<vector<vector<double>>>& outputGradients);
};


class PoolingLayer: public Layer{

public:
	int startRow,startCol,endRow,endCol;
	string poolingType;
    PoolingLayer (int inputChannels, int filterCount, int filterDimension, int padding = VALID, int stride = 1, string type = "Max",double learning_rate = 0.1);
    double max_finder2D(vector<vector<double>>mat,int focusRow,int focusCol,int filterDimension);
    double avg_finder2D(vector<vector<double>>mat,int focusRow,int focusCol,int filterDimension);
    void forwardPass(vector<vector<vector<double>>> &inputImage);
    void InitializeWeights();
    void updateWeights();
	void updateBiases();
    void un_max2D(vector<vector<double>>mat,int channel_no,int focusRow,int focusCol,double grad_val,int filterDimension);
    void backwardPass(vector<vector<vector<double>>>& outputGradients, int index, vector<Layer*>& Layers);
	void backwardPassBias(vector<vector<vector<double>>>& outputGradients);
	void backwardPassFirstLayer(vector<vector<vector<double>>>& outputGradients, int index, vector<Layer*>& Layers, vector<vector<vector<double>>>& inputImage);
};

class ConvLayer: public Layer{
    public:
        ConvLayer (int inputChannels, int filterCount, int filterDimension, int padding = VALID, int stride = 1, double learning_rate = 0.1);
        double  relu(double sum);
        double convolution(vector<vector<vector<double>>> &mat,int row,int col,int filterDimension,vector<vector<vector<double>>> &filters);
    	void forwardPass(vector<vector<vector<double>>> &inputImage);
    	void backwardPass(vector<vector<vector<double>>>& outputGradients, int index, vector<Layer*>& Layers);
    	void backwardPassFirstLayer(vector<vector<vector<double>>>& outputGradients, int index, vector<Layer*>& Layers, vector<vector<vector<double>>>& inputImage);
        void backwardPassBias(vector<vector<vector<double>>>&outputGradients);
        void updateBiases();
        void InitializeWeights();
        void updateWeights();
    

};

class CNNnet {
    public:
        vector<Layer*> Layers;
        vector<int> topology;	
        vector<vector<vector<double>>> inputImage;
        double learning_rate;
        vector<double>flattenedOutput;
		CNNnet();
    	CNNnet(vector<vector<int>>& networkTopology, vector<vector<vector<double>>>& inputImageData, double learning_rate = 0.1);
    	void forwardPass(vector<vector<vector<double>>>& inputImage);
    	void InitializeLayers();        	
        void backwardPass(vector<double>& gradientsOfANN);
        void flatten();
};
#endif
