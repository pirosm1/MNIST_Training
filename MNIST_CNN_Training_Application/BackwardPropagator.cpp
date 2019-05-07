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

	auto start = std::chrono::steady_clock::now();

	// Calculate newOutputLayerBases (b2)
	newOutputLayerBases = std::vector<float>(numberOfOutputs);
	for (int i = 0; i < numberOfOutputs; i++) {
		newOutputLayerBases[i] = outputLayerBases[i] - learningRate * (outputs[i] - truthBits[i]) * outputs[i] * (1 - outputs[i]);
	}
	auto end = std::chrono::steady_clock::now();
	std::cout << "New Output Layers Bases (b2) completed in: " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << " seconds" << std::endl << std::endl;

	start = std::chrono::steady_clock::now();
	// Calculate newOutputLayerWeights (v)
	newOutputLayerWeights = std::vector<std::vector<float>>(numberOfFullyConnectedNodes, std::vector<float>(numberOfOutputs));
	for (int i = 0; i < numberOfFullyConnectedNodes; i++)
		for (int j = 0; j < numberOfOutputs; j++) {
			newOutputLayerWeights[i][j] = outputLayerWeights[i][j] - learningRate * (outputs[j] - truthBits[j]) * (outputs[j] - outputs[j] * outputs[j]) * fullyConnectedLayer[i];
		}
	end = std::chrono::steady_clock::now();
	std::cout << "New Output Layer Weights (v) completed in: " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << " seconds" << std::endl << std::endl;

	start = std::chrono::steady_clock::now();
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
	end = std::chrono::steady_clock::now();
	std::cout << "New Fully Connected Layer Bases (b1) completed in: " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << " seconds" << std::endl << std::endl;

	start = std::chrono::steady_clock::now();
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
	end = std::chrono::steady_clock::now();
	std::cout << "New Fully Connected Layer Weights (u) in: " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << " seconds" << std::endl << std::endl;

	start = std::chrono::steady_clock::now();
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
	end = std::chrono::steady_clock::now();
	std::cout << "New Convolutional Layer Bases (b0) completed in: " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << " seconds" << std::endl << std::endl;

	std::vector<float> fImage(image.size());
	for (int i = 0; i < image.size(); i++)
		fImage[i] = (float)(unsigned(image[i])) / 255.0f;

	start = std::chrono::steady_clock::now();
	// Calculate newConvolutionalLayerWeights (w0)
	newConvolutionalLayerWeights = std::vector<std::vector<std::vector<float>>>(numberOfFeatures, std::vector<std::vector<float>>(sizeOfFilter, std::vector<float>(sizeOfFilter)));
	for (int i = 0; i < numberOfFeatures; i++) {
		for (int k = 0; k < sizeOfFilter; k++) {
			for (int m = 0; m < sizeOfFilter; m++) {
				float sum = 0.0f;
				for (int l = 0; l < numberOfMaxPoolNodes; l++) {
					int maxIndex = maxPoolingLayer[i][l].max;
					float max = maxPoolingLayer[i][l].values[maxIndex];
					int input = sizeOfInput * (2 * l / sizeOfMaxPool + (maxIndex / sizeOfPoolingWindow) + k) + (2 * (1 - l / sizeOfMaxPool * sizeOfMaxPool) + (maxIndex - sizeOfPoolingWindow * (maxIndex / sizeOfPoolingWindow) + m));

					float h0Derivative = max * (1.0f - max);
					float s0Derivative = fImage[input];

					for (int x = 0; x < numberOfFullyConnectedNodes; x++) {
						float y1Derivative = fullyConnectedLayer[x] * (1.0f - fullyConnectedLayer[x]);
						float r0Derivative = fullyConnectedLayerWeights[i][l][x];
						for (int j = 0; j < numberOfOutputs; j++) {
							float eDerivative = outputs[j] - truthBits[j];
							float zDerivative = outputs[j] * (1.0f - outputs[j]);
							float r1Derivative = outputLayerWeights[x][j];
							sum += eDerivative * zDerivative * r1Derivative * y1Derivative * r0Derivative * h0Derivative * s0Derivative;
						}
					}
				}
				newConvolutionalLayerWeights[i][k][m] = convolutionalLayerWeights[i][k][m] - learningRate * sum;
			}
		}
	}
	end = std::chrono::steady_clock::now();
	std::cout << "New Convolutional Layer Weights (w0) completed in: " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << " seconds" << std::endl << std::endl;
}
