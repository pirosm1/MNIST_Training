#include "pch.h"
#include "ForwardPropagator.h"



ForwardPropagator::ForwardPropagator() {}

ForwardPropagator::~ForwardPropagator() {}

void ForwardPropagator::setVerbose(bool v) {
	verbose = v;
}

void ForwardPropagator::setSizeOfInput(int size) {
	sizeOfInput = size;
}

void ForwardPropagator::setSizeOfFilter(int size) {
	sizeOfFilter = size;
}

void ForwardPropagator::setSizeOfFeature(int size) {
	sizeOfFeature = size;
}

void ForwardPropagator::setSizeOfMaxPool(int size) {
	sizeOfMaxPool = size;
	numberOfMaxPoolNodes = size * size;
}

void ForwardPropagator::setSizeOfPoolingWindow(int size) {
	sizeOfPoolingWindow = size;
}

void ForwardPropagator::setNumberOfFeatures(int n) {
	numberOfFeatures = n;
}

void ForwardPropagator::setNumberOfFullyConnectedNodes(int n) {
	numberOfFullyConnectedNodes = n;
}

void ForwardPropagator::setNumberOfOutputs(int n) {
	numberOfOutputs = n;
}

void ForwardPropagator::setConvolutionalLayerWeights(std::vector<std::vector<std::vector<float>>> weights) {
	convolutionalLayerWeights = weights;
}

void ForwardPropagator::setConvolutionalLayerBases(std::vector<float> bases) {
	convolutionalLayerBases = bases;
}

void ForwardPropagator::setFullyConnectedLayerWeights(std::vector<std::vector<std::vector<float>>> weights) {
	fullyConnectedLayerWeights = weights;
}

void ForwardPropagator::setFullyConnectedLayerBases(std::vector<float> bases) {
	outputLayerBases = bases;
}

void ForwardPropagator::setOutputLayerWeights(std::vector<std::vector<float>> weights) {
	outputLayerWeights = weights;
}

void ForwardPropagator::setOutputLayerBases(std::vector<float> bases) {
	outputLayerBases = bases;
}

std::vector<std::vector<std::vector<float>>> ForwardPropagator::getConvolutionalLayer() {
	return convolutionalLayer;
}

std::vector<std::vector<std::vector<float>>> ForwardPropagator::getConvolutionalLayerWeights() {
	return convolutionalLayerWeights;
}

std::vector<float> ForwardPropagator::getConvolutionalLayerBases() {
	return convolutionalLayerBases;
}

std::vector<std::vector<MAXPOOL>> ForwardPropagator::getMaxPoolingLayer() {
	return maxPoolingLayer;
}

std::vector<float> ForwardPropagator::getFullyConnectedLayer() {
	return fullyConnectedLayer;
}

std::vector<std::vector<std::vector<float>>> ForwardPropagator::getFullyConnectedLayerWeights() {
	return fullyConnectedLayerWeights;
}

std::vector<float> ForwardPropagator::getFullyConnectedLayerBases() {
	return fullyConnectedLayerBases;
}

std::vector<float> ForwardPropagator::getOutputLayer() {
	return outputLayer;
}

std::vector<std::vector<float>> ForwardPropagator::getOutputLayerWeights() {
	return outputLayerWeights;
}

std::vector<float> ForwardPropagator::getOutputLayerBases()
{
	return outputLayerBases;
}

void ForwardPropagator::randomInitialization() {
	// generate random seed
	srand((unsigned int)time(0));

	// convolutionalLayerWeights - 3 Dimensional weight array (6 layers x 5 pixels x 5 pixels) floats
	convolutionalLayerWeights = std::vector<std::vector<std::vector<float>>>(numberOfFeatures, std::vector<std::vector<float>>(sizeOfFilter, std::vector<float>(sizeOfFilter)));
	for (auto&& x : convolutionalLayerWeights)
		for (auto&& y : x)
			for (auto&& z : y)
				z = random();

	// convolutionalLayerBases - (6 layers) floats
	convolutionalLayerBases = std::vector<float>(numberOfFeatures);
	for (auto&& x : convolutionalLayerBases)
		x = random();

	// fullyConnectedLayerWeights - 3 Dimensional weight array (6 layers x 144 max-pool nodes x 45 fullyConnected Nodes) floats
	fullyConnectedLayerWeights = std::vector<std::vector<std::vector<float>>>(numberOfFeatures, std::vector<std::vector<float>>(numberOfMaxPoolNodes, std::vector<float>(numberOfFullyConnectedNodes)));
	for (auto&& x : fullyConnectedLayerWeights)
		for (auto&& y : x)
			for (auto&& z : y)
				z = random();

	// fullyConnectedLayerBases - (45 fully connected nodes) floats
	fullyConnectedLayerBases = std::vector<float>(numberOfFullyConnectedNodes);
	for (auto&& x : fullyConnectedLayerBases)
		x = random();

	// outputWeights - 2 Dimensional Weight array (45 fully connected nodes x 10 outputs)
	outputLayerWeights = std::vector<std::vector<float>>(numberOfFullyConnectedNodes, std::vector<float>(numberOfOutputs));
	for (auto&& x : outputLayerWeights)
		for (auto&& y : x)
			y = random();

	// outputBases - (10 outputs) floats
	outputLayerBases = std::vector<float>(numberOfOutputs);
	for (auto&& x : outputLayerBases)
		x = random();

	// print to weight file
	std::ofstream weightFile("randomized_weight_values.txt");
	if (weightFile.is_open()) {
		printWeightValues(weightFile, false);
		weightFile.close();
	}
	if (verbose)
		printWeightValues(std::cout, true);
}

std::vector<float> ForwardPropagator::forwardPropagation(std::vector<uint8_t> image) {
	// Populate convolutionalLayer - 3 Dimensional sigmoid-activated sum array 6x24x24
	convolutionalLayer = std::vector<std::vector<std::vector<float>>>(numberOfFeatures, std::vector<std::vector<float>>(sizeOfFeature, std::vector<float>(sizeOfFeature)));
	// for each feature
	for (int i = 0; i < numberOfFeatures; i++) {
		// for filter 'window' in the input
		for (int y = 0; y < sizeOfFeature; y++) {
			for (int x = 0; x < sizeOfFeature; x++) {

				// compute the sum of the filter * corresponding weight
				float sum = 0.0f;

				for (int yOffset = 0; yOffset < sizeOfFilter; yOffset++) {
					for (int xOffset = 0; xOffset < sizeOfFilter; xOffset++) {
						sum += (float)unsigned(image[sizeOfInput*(y + yOffset) + x + xOffset]) / 255 * convolutionalLayerWeights[i][yOffset][xOffset];
					}
				}

				sum += convolutionalLayerBases[i];
				convolutionalLayer[i][y][x] = fastSigmoid(sum);
			}
		}
	}

	// Populate maxPoolingLayer - 2 Dimensional max array 6x144
	maxPoolingLayer = std::vector<std::vector<MAXPOOL>>(numberOfFeatures, std::vector<MAXPOOL>(numberOfMaxPoolNodes));
	// for each feature
	for (int i = 0; i < numberOfFeatures; i++) {
		// for each 2x2 pool in the convolutional layer
		int maxPoolNode = 0;
		for (int y = 0; y < sizeOfFeature; y += sizeOfPoolingWindow) {
			for (int x = 0; x < sizeOfFeature; x += sizeOfPoolingWindow) {
				int count = 0;
				for (int yOffset = 0; yOffset < sizeOfPoolingWindow; yOffset++) {
					for (int xOffset = 0; xOffset < sizeOfPoolingWindow; xOffset++) {
						maxPoolingLayer[i][maxPoolNode].values[count] = convolutionalLayer[i][y + yOffset][x + xOffset];
						count++;
					}
				}
				maxPoolingLayer[i][maxPoolNode].max = max(maxPoolingLayer[i][maxPoolNode].values);
				maxPoolNode++;
			}
		}
	}

	// Populate each fully connected layer
	fullyConnectedLayer = std::vector<float>(numberOfFullyConnectedNodes);
	// for each fullyConnectedNode
	for (int i = 0; i < numberOfFullyConnectedNodes; i++) {
		float sum = 0.0f;
		// for each maxPoolLayer (features)
		for (int j = 0; j < numberOfFeatures; j++) {
			// Add up every maxPoolNode * fullyConnectedWeight
			for (int k = 0; k < numberOfMaxPoolNodes; k++)
				sum += maxPoolingLayer[j][k].values[maxPoolingLayer[j][k].max] * fullyConnectedLayerWeights[j][k][i];
		}
		// Add the fullyConnectedLayerBase
		sum += fullyConnectedLayerBases[i];
		fullyConnectedLayer[i] = fastSigmoid(sum);
	}

	// Populate each output layer
	outputLayer = std::vector<float>(numberOfOutputs);
	// for each outputLayer
	for (int i = 0; i < numberOfOutputs; i++) {
		float sum = 0.0f;
		// for each fullyConnectedLayer
		for (int j = 0; j < numberOfFullyConnectedNodes; j++) {
			sum += fullyConnectedLayer[j] * outputLayerWeights[j][i];
		}
		// Add the OutputLayer's base
		sum += outputLayerBases[i];
		outputLayer[i] = fastSigmoid(sum);
	}

	return outputLayer;
}

float ForwardPropagator::random() {
	return (float)rand() / RAND_MAX - 0.5f;
}

void ForwardPropagator::printWeightValues(std::ostream &output, bool verbose = false)
{
	if (verbose) {
		output << "Size of Input: " << sizeOfInput << "x" << sizeOfInput << std::endl;
		output << "Size of Convolution Filter: " << sizeOfFilter << "x" << sizeOfFilter << std::endl;
		output << "Size of Convolution Feature: " << sizeOfFeature << "x" << sizeOfFeature << std::endl;
		output << "Size of Max-Pool: " << sizeOfMaxPool << "x" << sizeOfMaxPool << std::endl;
		output << "Number of Features: " << numberOfFeatures << std::endl;
		output << "Number of Fully Connected Nodes: " << numberOfFullyConnectedNodes << std::endl;
		output << "Number of Outputs: " << numberOfOutputs << std::endl;
		output << std::endl;
	}

	// Print out weight values per feature
	if (verbose)
		output << "Convolution Weights" << std::endl;
	else {
		output << numberOfFeatures << std::endl;
		output << sizeOfFilter << std::endl;
	}
	for (int i = 0; i < convolutionalLayerWeights.size(); i++) {
		if (verbose)
			output << "Feature " << i << std::endl;
		for (auto&& y : convolutionalLayerWeights[i]) {
			for (auto&& z : y)
				output << z << " ";
			output << std::endl;
		}
	}

	// Print out convolution base values
	if (verbose)
		output << "Convolution Bases" << std::endl;
	for (int i = 0; i < numberOfFeatures; i++) {
		if (verbose)
			output << "Feature " << i << std::endl;
		output << convolutionalLayerBases[i] << std::endl;
	}

	// Print out fully connected layer weights
	if (verbose)
		output << "Fully Connected Layer Weights" << std::endl;
	else {
		output << numberOfMaxPoolNodes << std::endl;
		output << numberOfFullyConnectedNodes << std::endl;
	}
	for (int i = 0; i < numberOfFeatures; i++) {
		if (verbose)
			output << "Feature " << i << std::endl;
		for (auto&& y : fullyConnectedLayerWeights[i]) {
			for (auto&& x : y) {
				output << x << " ";
			}
			output << std::endl;
		}
	}

	// Print out fully connected layer bases
	if (verbose)
		output << "Fully Connected Layer Bases" << std::endl;
	for (int i = 0; i < numberOfFullyConnectedNodes; i++) {
		if (verbose)
			output << "FullyConnectedNode " << i << std::endl;
		output << fullyConnectedLayerBases[i] << std::endl;
	}

	// Print out output weights
	if (verbose)
		output << "Output Weights" << std::endl;
	else
		output << numberOfOutputs << std::endl;
	for (auto&& y : outputLayerWeights)
		for (auto&& x : y)
			output << x << " ";
	output << std::endl;

	// Print out outputs bases
	if (verbose)
		output << "Output Bases" << std::endl;
	for (int i = 0; i < numberOfOutputs; i++) {
		if (verbose)
			output << "Feature " << i << std::endl;
		output << outputLayerBases[i] << std::endl;
	}
}