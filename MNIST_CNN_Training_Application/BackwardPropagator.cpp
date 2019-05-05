#include "BackwardPropagator.h"


BackwardPropagator::BackwardPropagator() {}

BackwardPropagator::~BackwardPropagator() {}

void BackwardPropagator::setLearningRate(float l) {
	learningRate = l;
}

void BackwardPropagator::setSizeOfInput(int size) {
	sizeOfInput = size;
}

void BackwardPropagator::setSizeOfFilter(int size) {
	sizeOfFilter = size;
}

void BackwardPropagator::setSizeOfFeature(int size) {
	sizeOfFeature = size;
}

void BackwardPropagator::setSizeOfMaxPool(int size) {
	sizeOfMaxPool = size;
	numberOfMaxPoolNodes = size * size;
}

void BackwardPropagator::setSizeOfPoolingWindow(int size) {
	sizeOfPoolingWindow = size;
}

void BackwardPropagator::setNumberOfFeatures(int n) {
	numberOfFeatures = n;
}

void BackwardPropagator::setNumberOfFullyConnectedNodes(int n) {
	numberOfFullyConnectedNodes = n;
}

void BackwardPropagator::setNumberOfOutputs(int n) {
	numberOfOutputs = n;
}

void BackwardPropagator::setConvolutionalLayerWeights(std::vector<std::vector<std::vector<float>>> weights) {
	convolutionalLayerWeights = weights;
}

void BackwardPropagator::setConvolutionalLayerBases(std::vector<float> bases) {
	convolutionalLayerBases = bases;
}

void BackwardPropagator::setMaxPoolingLayer(std::vector<std::vector<MAXPOOL>> v) {
	maxPoolingLayer = v;
}

void BackwardPropagator::setFullyConnectedLayer(std::vector<float> v) {
	fullyConnectedLayer = v;
}

void BackwardPropagator::setFullyConnectedLayerWeights(std::vector<std::vector<std::vector<float>>> weights) {
	fullyConnectedLayerWeights = weights;
}

void BackwardPropagator::setFullyConnectedLayerBases(std::vector<float> bases) {
	fullyConnectedLayerBases = bases;
}

void BackwardPropagator::setOutputLayerWeights(std::vector<std::vector<float>> weights) {
	outputLayerWeights = weights;
}

void BackwardPropagator::setOutputLayerBases(std::vector<float> bases) {
	outputLayerBases = bases;
}

std::vector<std::vector<std::vector<float>>> BackwardPropagator::getNewConvolutionalLayerWeights() {
	return newConvolutionalLayerWeights;
}

std::vector<float> BackwardPropagator::getNewConvolutionalLayerBases() {
	return newConvolutionalLayerBases;
}

std::vector<std::vector<std::vector<float>>> BackwardPropagator::getNewFullyConnectedLayerWeights() {
	return newFullyConnectedLayerWeights;
}

std::vector<float> BackwardPropagator::getNewFullyConnectedLayerBases() {
	return newFullyConnectedLayerBases;
}

std::vector<std::vector<float>> BackwardPropagator::getNewOutputLayerWeights() {
	return newOutputLayerWeights;
}

std::vector<float> BackwardPropagator::getNewOutputLayerBases() {
	return newOutputLayerBases;
}

void BackwardPropagator::backwardPropagation(std::vector<uint8_t> image, std::vector<float> outputs, int truth) {
	std::bitset<10> truthBits = makeBitset(truth);

	// Calculate newOutputLayerBases (b2)
	newOutputLayerBases = std::vector<float>(numberOfOutputs);
	for (int i = 0; i < numberOfOutputs; i++) {
		newOutputLayerBases[i] = outputLayerBases[i] - learningRate * (outputs[i] - truthBits[i]) * (outputs[i] - outputs[i] * outputs[i]);
	}

	// Calculate newOutputLayerWeights (v)
	newOutputLayerWeights = std::vector<std::vector<float>>(numberOfFullyConnectedNodes, std::vector<float>(numberOfOutputs));
	for (int i = 0; i < numberOfFullyConnectedNodes; i++)
		for (int j = 0; j < numberOfOutputs; j++) {
			newOutputLayerWeights[i][j] = outputLayerWeights[i][j] - learningRate * (outputs[j] - truthBits[j]) * (outputs[j] - outputs[j] * outputs[j]) * fullyConnectedLayer[i];
		}

	// Calculate newFullyConnectedLayerBases (b1)
	newFullyConnectedLayerBases = std::vector<float>(numberOfFullyConnectedNodes);
	for (int i = 0; i < numberOfFullyConnectedNodes; i++) {
		float sum = 0.0f;
		for (int j = 0; j < numberOfOutputs; j++) {
			// (z - t) * (z - z*z) * v
			sum += (outputs[j] - truthBits[j]) * (outputs[j] - outputs[j] * outputs[j]) * outputLayerWeights[i][j];
		}

		newFullyConnectedLayerBases[i] = fullyConnectedLayerBases[i] - learningRate * sum * (fullyConnectedLayer[i] - fullyConnectedLayer[i] * fullyConnectedLayer[i]);
	}

	// Calculate newFullyConnectedLayerWeights (u)
	newFullyConnectedLayerWeights = std::vector<std::vector<std::vector<float>>>(numberOfFeatures, std::vector<std::vector<float>>(numberOfMaxPoolNodes, std::vector<float>(numberOfFullyConnectedNodes)));
	for (int i = 0; i < numberOfFeatures; i++) {
		for (int j = 0; j < numberOfMaxPoolNodes; j++) {
			for (int k = 0; k < numberOfFullyConnectedNodes; k++) {
				float sum = 0.0f;
				for (int l = 0; l < numberOfOutputs; l++) {
					sum += (outputs[l] - truthBits[l]) * (outputs[l] - outputs[l] * outputs[l]) * outputLayerWeights[k][l];
				}
				newFullyConnectedLayerWeights[i][j][k] = fullyConnectedLayerWeights[i][j][k] - learningRate * sum * (fullyConnectedLayer[k] - fullyConnectedLayer[k] * fullyConnectedLayer[k]) * maxPoolingLayer[i][j].values[maxPoolingLayer[i][j].max];
			}
		}
	}

	// Calculate newConvolutionalLayerBases (b0)
	newConvolutionalLayerBases = std::vector<float>(numberOfFeatures);
	for (int i = 0; i < numberOfFeatures; i++) {
		float sum = 0.0f;
		for (int j = 0; j < numberOfOutputs; j++) {
			for (int k = 0; k < numberOfFullyConnectedNodes; k++) {
				for (int l = 0; l < numberOfMaxPoolNodes; l++) {
					float max = maxPoolingLayer[i][l].values[maxPoolingLayer[i][l].max];
					// (z - t) * (z - z^2) * v * (y - y^2) * u * h
					sum += (outputs[j] - truthBits[j]) * (outputs[j] - outputs[j] * outputs[j]) * outputLayerWeights[k][j] * (fullyConnectedLayer[k] - fullyConnectedLayer[k] * fullyConnectedLayer[k]) * fullyConnectedLayerWeights[i][l][k] * (max - max * max);
				}
			}
		}
		newConvolutionalLayerBases[i] = convolutionalLayerBases[i] - learningRate * sum;
	}

	// Calculate newConvolutionalLayerWeights (w0)
	newConvolutionalLayerWeights = std::vector<std::vector<std::vector<float>>>(numberOfFeatures, std::vector<std::vector<float>>(sizeOfFilter, std::vector<float>(sizeOfFilter)));
	for (int i = 0; i < numberOfFeatures; i++) {
		for (int k = 0; k < sizeOfFilter; k++) {
			for (int m = 0; m < sizeOfFilter; m++) {
				float sum = 0.0f;
				for (int j = 0; j < numberOfOutputs; j++) {
					for (int x = 0; x < numberOfFullyConnectedNodes; x++) {
						for (int l = 0; l < numberOfMaxPoolNodes; l++) {
							int maxIndex = maxPoolingLayer[i][l].max;
							//int rowInMaxPoolingLayer = l / sizeOfMaxPool;
							//int colInMaxPoolingLayer = l - rowInMaxPoolingLayer * sizeOfMaxPool;
							//int rowOffset = maxIndex / sizeOfPoolingWindow;
							//int colOffset = maxIndex - sizeOfPoolingWindow * rowOffset;
							//int rowInInput = 2 * rowInMaxPoolingLayer + rowOffset + k;
							//int colInInput = 2 * colInMaxPoolingLayer + colOffset + m;
							int input = sizeOfInput * (2 * l / sizeOfMaxPool + (maxIndex / sizeOfPoolingWindow) + k) + (2 * (1 - l / sizeOfMaxPool * sizeOfMaxPool) + (maxIndex - sizeOfPoolingWindow * (maxIndex / sizeOfPoolingWindow) + m));
							float max = maxPoolingLayer[i][l].values[maxIndex];

							sum += (outputs[j] - truthBits[j]) * (outputs[j] - outputs[j] * outputs[j]) * outputLayerWeights[x][j] * (fullyConnectedLayer[x] - fullyConnectedLayer[x] * fullyConnectedLayer[x]) * fullyConnectedLayerWeights[i][l][x] * (max - max * max) * unsigned(image[input])/255;
						}
					}
				}
				newConvolutionalLayerWeights[i][k][m] = convolutionalLayerWeights[i][k][m] - learningRate * sum;
			}
		}
	}
}
