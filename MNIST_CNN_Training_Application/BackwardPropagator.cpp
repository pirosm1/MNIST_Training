#include "pch.h"
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

void BackwardPropagator::backwardPropagation(std::vector<uint8_t> image, std::vector<float> outputs, int truth, bool verbose = false) {
	std::bitset<10> truthBits = makeBitset(truth);

	auto start = std::chrono::steady_clock::now();
	auto end = std::chrono::steady_clock::now();
	if (verbose)
		start = std::chrono::steady_clock::now();
	// Calculate newOutputLayerBases (b2)
	newOutputLayerBases = std::vector<float>(numberOfOutputs);
	for (int i = 0; i < numberOfOutputs; i++) {
		newOutputLayerBases[i] = outputLayerBases[i] - learningRate * (outputs[i] - truthBits[i]) * outputs[i] * (1 - outputs[i]);
	}
	if (verbose) {
		end = std::chrono::steady_clock::now();
		std::cout << "New Output Layers Bases (b2) completed in: " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << " seconds" << std::endl << std::endl;
	}

	if (verbose)
		start = std::chrono::steady_clock::now();
	// Calculate newOutputLayerWeights (v)
	newOutputLayerWeights = std::vector<std::vector<float>>(numberOfFullyConnectedNodes, std::vector<float>(numberOfOutputs));
	for (int i = 0; i < numberOfFullyConnectedNodes; i++)
		for (int j = 0; j < numberOfOutputs; j++) {
			newOutputLayerWeights[i][j] = outputLayerWeights[i][j] - learningRate * (outputs[j] - truthBits[j]) * (outputs[j] - outputs[j] * outputs[j]) * fullyConnectedLayer[i];
		}
	if (verbose) {
		end = std::chrono::steady_clock::now();
		std::cout << "New Output Layer Weights (v) completed in: " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << " seconds" << std::endl << std::endl;
	}

	if (verbose)
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
	if (verbose) {
		end = std::chrono::steady_clock::now();
		std::cout << "New Fully Connected Layer Bases (b1) completed in: " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << " seconds" << std::endl << std::endl;
	}

	if (verbose)
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
	if (verbose) {
		end = std::chrono::steady_clock::now();
		std::cout << "New Fully Connected Layer Weights (u) in: " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << " seconds" << std::endl << std::endl;
	}

	if (verbose)
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
	if (verbose) {
		end = std::chrono::steady_clock::now();
		std::cout << "New Convolutional Layer Bases (b0) completed in: " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << " seconds" << std::endl << std::endl;
	}

	// Normalize the image before hand
	std::vector<float> fImage(image.size());
	for (int i = 0; i < image.size(); i++)
		fImage[i] = (float)(unsigned(image[i])) / 255.0f;

	if (verbose)
		start = std::chrono::steady_clock::now();
	// Calculate newConvolutionalLayerWeights (w0)
	newConvolutionalLayerWeights = std::vector<std::vector<std::vector<float>>>(numberOfFeatures, std::vector<std::vector<float>>(sizeOfFilter, std::vector<float>(sizeOfFilter)));
	for (int k = 0; k < sizeOfFilter; k++) {
		for (int m = 0; m < sizeOfFilter; m++) {
			float sum0 = 0.0f;
			float sum1 = 0.0f;
			float sum2 = 0.0f;
			float sum3 = 0.0f;
			float sum4 = 0.0f;
			float sum5 = 0.0f;
			float sum6 = 0.0f;
			float sum7 = 0.0f;
			for (int l = 0; l < numberOfMaxPoolNodes; l++) {
				int maxIndex0 = maxPoolingLayer[0][l].max;
				int maxIndex1 = maxPoolingLayer[1][l].max;
				int maxIndex2 = maxPoolingLayer[2][l].max;
				int maxIndex3 = maxPoolingLayer[3][l].max;
				int maxIndex4 = maxPoolingLayer[4][l].max;
				int maxIndex5 = maxPoolingLayer[5][l].max;
				int maxIndex6 = maxPoolingLayer[6][l].max;
				int maxIndex7 = maxPoolingLayer[7][l].max;
				float max0 = maxPoolingLayer[0][l].values[maxIndex0];
				float max1 = maxPoolingLayer[1][l].values[maxIndex1];
				float max2 = maxPoolingLayer[2][l].values[maxIndex2];
				float max3 = maxPoolingLayer[3][l].values[maxIndex3];
				float max4 = maxPoolingLayer[4][l].values[maxIndex4];
				float max5 = maxPoolingLayer[5][l].values[maxIndex5];
				float max6 = maxPoolingLayer[6][l].values[maxIndex6];
				float max7 = maxPoolingLayer[7][l].values[maxIndex7];

				int input0 = sizeOfInput * (2 * l / sizeOfMaxPool + (maxIndex0 / sizeOfPoolingWindow) + k) + (2 * (1 - l / sizeOfMaxPool * sizeOfMaxPool) + (maxIndex0 - sizeOfPoolingWindow * (maxIndex0 / sizeOfPoolingWindow) + m));
				int input1 = sizeOfInput * (2 * l / sizeOfMaxPool + (maxIndex1 / sizeOfPoolingWindow) + k) + (2 * (1 - l / sizeOfMaxPool * sizeOfMaxPool) + (maxIndex1 - sizeOfPoolingWindow * (maxIndex1 / sizeOfPoolingWindow) + m));
				int input2 = sizeOfInput * (2 * l / sizeOfMaxPool + (maxIndex2 / sizeOfPoolingWindow) + k) + (2 * (1 - l / sizeOfMaxPool * sizeOfMaxPool) + (maxIndex2 - sizeOfPoolingWindow * (maxIndex2 / sizeOfPoolingWindow) + m));
				int input3 = sizeOfInput * (2 * l / sizeOfMaxPool + (maxIndex3 / sizeOfPoolingWindow) + k) + (2 * (1 - l / sizeOfMaxPool * sizeOfMaxPool) + (maxIndex3 - sizeOfPoolingWindow * (maxIndex3 / sizeOfPoolingWindow) + m));
				int input4 = sizeOfInput * (2 * l / sizeOfMaxPool + (maxIndex4 / sizeOfPoolingWindow) + k) + (2 * (1 - l / sizeOfMaxPool * sizeOfMaxPool) + (maxIndex4 - sizeOfPoolingWindow * (maxIndex4 / sizeOfPoolingWindow) + m));
				int input5 = sizeOfInput * (2 * l / sizeOfMaxPool + (maxIndex5 / sizeOfPoolingWindow) + k) + (2 * (1 - l / sizeOfMaxPool * sizeOfMaxPool) + (maxIndex5 - sizeOfPoolingWindow * (maxIndex5 / sizeOfPoolingWindow) + m));
				int input6 = sizeOfInput * (2 * l / sizeOfMaxPool + (maxIndex6 / sizeOfPoolingWindow) + k) + (2 * (1 - l / sizeOfMaxPool * sizeOfMaxPool) + (maxIndex6 - sizeOfPoolingWindow * (maxIndex6 / sizeOfPoolingWindow) + m));
				int input7 = sizeOfInput * (2 * l / sizeOfMaxPool + (maxIndex7 / sizeOfPoolingWindow) + k) + (2 * (1 - l / sizeOfMaxPool * sizeOfMaxPool) + (maxIndex7 - sizeOfPoolingWindow * (maxIndex7 / sizeOfPoolingWindow) + m));


				float h0Derivative0 = max0 * (1.0f - max0);
				float h0Derivative1 = max1 * (1.0f - max1);
				float h0Derivative2 = max2 * (1.0f - max2);
				float h0Derivative3 = max3 * (1.0f - max3);
				float h0Derivative4 = max4 * (1.0f - max4);
				float h0Derivative5 = max5 * (1.0f - max5);
				float h0Derivative6 = max6 * (1.0f - max6);
				float h0Derivative7 = max7 * (1.0f - max7);

				float s0Derivative0 = fImage[input0];
				float s0Derivative1 = fImage[input1];
				float s0Derivative2 = fImage[input2];
				float s0Derivative3 = fImage[input3];
				float s0Derivative4 = fImage[input4];
				float s0Derivative5 = fImage[input5];
				float s0Derivative6 = fImage[input6];
				float s0Derivative7 = fImage[input7];

				for (int x = 0; x < numberOfFullyConnectedNodes; x++) {
					float y1Derivative = fullyConnectedLayer[x] * (1.0f - fullyConnectedLayer[x]);

					float r0Derivative0 = fullyConnectedLayerWeights[0][l][x];
					float r0Derivative1 = fullyConnectedLayerWeights[1][l][x];
					float r0Derivative2 = fullyConnectedLayerWeights[2][l][x];
					float r0Derivative3 = fullyConnectedLayerWeights[3][l][x];
					float r0Derivative4 = fullyConnectedLayerWeights[4][l][x];
					float r0Derivative5 = fullyConnectedLayerWeights[5][l][x];
					float r0Derivative6 = fullyConnectedLayerWeights[6][l][x];
					float r0Derivative7 = fullyConnectedLayerWeights[7][l][x];
					for (int j = 0; j < numberOfOutputs; j++) {
						float eDerivative = outputs[j] - truthBits[j];
						float zDerivative = outputs[j] * (1.0f - outputs[j]);
						float r1Derivative = outputLayerWeights[x][j];
						sum0 += eDerivative * zDerivative * r1Derivative * y1Derivative * r0Derivative0 * h0Derivative0 * s0Derivative0;
						sum1 += eDerivative * zDerivative * r1Derivative * y1Derivative * r0Derivative1 * h0Derivative1 * s0Derivative1;
						sum2 += eDerivative * zDerivative * r1Derivative * y1Derivative * r0Derivative2 * h0Derivative2 * s0Derivative2;
						sum3 += eDerivative * zDerivative * r1Derivative * y1Derivative * r0Derivative3 * h0Derivative3 * s0Derivative3;
						sum4 += eDerivative * zDerivative * r1Derivative * y1Derivative * r0Derivative4 * h0Derivative4 * s0Derivative4;
						sum5 += eDerivative * zDerivative * r1Derivative * y1Derivative * r0Derivative5 * h0Derivative5 * s0Derivative5;
						sum6 += eDerivative * zDerivative * r1Derivative * y1Derivative * r0Derivative6 * h0Derivative6 * s0Derivative6;
						sum7 += eDerivative * zDerivative * r1Derivative * y1Derivative * r0Derivative7 * h0Derivative7 * s0Derivative7;
					}
				}
			}
			newConvolutionalLayerWeights[0][k][m] = convolutionalLayerWeights[0][k][m] - learningRate * sum0;
			newConvolutionalLayerWeights[1][k][m] = convolutionalLayerWeights[1][k][m] - learningRate * sum1;
			newConvolutionalLayerWeights[2][k][m] = convolutionalLayerWeights[2][k][m] - learningRate * sum2;
			newConvolutionalLayerWeights[3][k][m] = convolutionalLayerWeights[3][k][m] - learningRate * sum3;
			newConvolutionalLayerWeights[4][k][m] = convolutionalLayerWeights[4][k][m] - learningRate * sum4;
			newConvolutionalLayerWeights[5][k][m] = convolutionalLayerWeights[5][k][m] - learningRate * sum5;
			newConvolutionalLayerWeights[6][k][m] = convolutionalLayerWeights[6][k][m] - learningRate * sum6;
			newConvolutionalLayerWeights[7][k][m] = convolutionalLayerWeights[7][k][m] - learningRate * sum7;
		}
	}
	if (verbose) {
		end = std::chrono::steady_clock::now();
		std::cout << "New Convolutional Layer Weights (w0) completed in: " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << " seconds" << std::endl << std::endl;
	}
}
