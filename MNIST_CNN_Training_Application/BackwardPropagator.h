#pragma once
#include <chrono>
#include <bitset>
#include <iostream>
#include <vector>

// #include "maxpool.h"

#ifndef MAXPOOL_H
#define MAXPOOL_H

struct MAXPOOL {
	int max;
	float values[4];
};

#endif // !MAXPOOL_H

class BackwardPropagator
{
public:
	BackwardPropagator();
	~BackwardPropagator();

	// Setters
	void setLearningRate(float l);
	void setSizeOfInput(int size);
	void setSizeOfFilter(int size);
	void setSizeOfFeature(int size);
	void setSizeOfMaxPool(int size);
	void setSizeOfPoolingWindow(int size);
	void setNumberOfFeatures(int n);
	void setNumberOfFullyConnectedNodes(int n);
	void setNumberOfOutputs(int n);
	void setConvolutionalLayerWeights(std::vector<std::vector<std::vector<float>>> weights);
	void setConvolutionalLayerBases(std::vector<float> bases);
	void setMaxPoolingLayer(std::vector<std::vector<MAXPOOL>> v);
	void setFullyConnectedLayer(std::vector<float> v);
	void setFullyConnectedLayerWeights(std::vector<std::vector<std::vector<float>>> weights);
	void setFullyConnectedLayerBases(std::vector<float> bases);
	void setOutputLayerWeights(std::vector<std::vector<float>> weights);
	void setOutputLayerBases(std::vector<float>);

	// Getters
	std::vector<std::vector<std::vector<float>>> getNewConvolutionalLayerWeights();
	std::vector<float> getNewConvolutionalLayerBases();
	std::vector<std::vector<std::vector<float>>> getNewFullyConnectedLayerWeights();
	std::vector<float> getNewFullyConnectedLayerBases();
	std::vector<std::vector<float>> getNewOutputLayerWeights();
	std::vector<float> getNewOutputLayerBases();

	void backwardPropagation(std::vector<uint8_t> image, std::vector<float>, int truth);

private:
	float learningRate;
	int sizeOfInput;
	int sizeOfFilter;
	int sizeOfFeature;
	int sizeOfMaxPool;
	int sizeOfPoolingWindow;
	int numberOfMaxPoolNodes;
	int numberOfFeatures;
	int numberOfFullyConnectedNodes;
	int numberOfOutputs;

	// Old Weights/Bases
	std::vector<std::vector<std::vector<float>>> convolutionalLayerWeights;
	std::vector<float> convolutionalLayerBases;
	std::vector<std::vector<MAXPOOL>> maxPoolingLayer;
	std::vector<float> fullyConnectedLayer;
	std::vector<std::vector<std::vector<float>>> fullyConnectedLayerWeights;
	std::vector<float> fullyConnectedLayerBases;

	std::vector<std::vector<float>> outputLayerWeights;
	std::vector<float> outputLayerBases;

	// New Weights/Bases
	std::vector<std::vector<std::vector<float>>> newConvolutionalLayerWeights;
	std::vector<float> newConvolutionalLayerBases;

	std::vector<std::vector<std::vector<float>>> newFullyConnectedLayerWeights;
	std::vector<float> newFullyConnectedLayerBases;

	std::vector<std::vector<float>> newOutputLayerWeights;
	std::vector<float> newOutputLayerBases;

	inline std::bitset<10> makeBitset(int n) {
		std::bitset<10> bits(0);

		bits.set(n);

		return bits;
	}
};

