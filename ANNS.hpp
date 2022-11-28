#ifndef ANNS_HPP
#define ANNS_HPP

#include<iostream>
#include<bits/stdc++.h>
#include<random>
#include<time.h>
#include<cstdlib>
#include<cmath>
using namespace std;



class Neuron{
    public:
        double output;
        vector<double> weights;
        double gradient;

        Neuron(double output);
        void initializeWeights(int weightSize,int n_index);
        void forwardPassFirstLayer(vector<double>& input, int n_index);
        double sigmoidActivation (double x);
        void forwardPassSigmoid(vector<Neuron>& prevLayer, int n_index);
        double forwardPassLastLayer(vector<Neuron>& prevLayer, int n_index);
        void backwardPass();
};

typedef vector<Neuron> Layer;

class NeuralNet{
    public:
        vector<Layer>Layers;
        double eta;

        NeuralNet(vector<int> topology,double eta);
        void initializeWeights();
        void forwardPass(vector<double>& input);
        void backwardPass(vector<double>& trueOutput);


};
#endif