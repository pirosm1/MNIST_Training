#pragma once
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <vector>
#include <time.h>

// #include "maxpool.h"

#ifndef MAXPOOL_H
#define MAXPOOL_H

struct MAXPOOL {
	int max;
	float values[4];
};

#endif // !MAXPOOL_H

class ForwardPropagator
{
public:
	ForwardPropagator();
	~ForwardPropagator();

	// Setters
	void setVerbose(bool v);
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
	void setFullyConnectedLayerWeights(std::vector<std::vector<std::vector<float>>> weights);
	void setFullyConnectedLayerBases(std::vector<float> bases);
	void setOutputLayerWeights(std::vector<std::vector<float>> weights);
	void setOutputLayerBases(std::vector<float>);

	// Getters
	std::vector<std::vector<std::vector<float>>> getConvolutionalLayer();
	std::vector<std::vector<std::vector<float>>> getConvolutionalLayerWeights();
	std::vector<float> getConvolutionalLayerBases();
	std::vector<std::vector<MAXPOOL>> getMaxPoolingLayer();
	std::vector<float> getFullyConnectedLayer();
	std::vector<std::vector<std::vector<float>>> getFullyConnectedLayerWeights();
	std::vector<float> getFullyConnectedLayerBases();
	std::vector<float> getOutputLayer();
	std::vector<std::vector<float>> getOutputLayerWeights();
	std::vector<float> getOutputLayerBases();

	std::vector<float> forwardPropagation(std::vector<uint8_t> image);

	void printWeightValues(std::ostream& output, bool verbose);

	void randomInitialization();
private:
	bool verbose = false;
	int sizeOfInput;
	int sizeOfFilter;
	int sizeOfFeature;
	int sizeOfMaxPool;
	int sizeOfPoolingWindow;
	int numberOfMaxPoolNodes;
	int numberOfFeatures;
	int numberOfFullyConnectedNodes;
	int numberOfOutputs;

	std::vector<std::vector<std::vector<float>>> convolutionalLayer;
	std::vector<std::vector<std::vector<float>>> convolutionalLayerWeights;
	std::vector<float> convolutionalLayerBases;
	std::vector<std::vector<MAXPOOL>> maxPoolingLayer;
	std::vector<float> fullyConnectedLayer;
	std::vector<std::vector<std::vector<float>>> fullyConnectedLayerWeights;
	std::vector<float> fullyConnectedLayerBases;
	std::vector<float> outputLayer;
	std::vector<std::vector<float>> outputLayerWeights;
	std::vector<float> outputLayerBases;

	float random();

	inline float fastSigmoid(float value) {
		return 1.0f / (1.0f + exp(-value));
	}

	inline int max(float values[]) {
		float max = values[0];
		int maxIndex = 0;

		for (int i = 1; i < sizeof(values) / sizeof(*values); i++) {
			if (values[i] > max) {
				max = values[i];
				maxIndex = i;
			}
		}

		return maxIndex;
	}
};

