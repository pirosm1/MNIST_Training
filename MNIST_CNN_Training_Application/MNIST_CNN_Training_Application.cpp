// MNIST_CNN_Training_Application.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include <chrono>
#include <iostream>
#include <fstream>
#include <string>

#include "MNISTReader.h"
#include "ForwardPropagator.h"
#include "BackwardPropagator.h"

void printUsage();
void printTestImage(std::vector<uint8_t> image);
int maxIndex(std::vector<float> values);

int main(int argc, char *argv[])
{
	std::ifstream trainingImageFile, trainingLabelFile;

	if (argc != 3) {
		printUsage();

		return 1;
	}

	const std::string trainingImageFileName = argv[1];
	const std::string trainingLabelFileName = argv[2];
	auto reader = std::make_unique<MNISTReader>(trainingImageFileName, trainingLabelFileName);

	// read images into memoery
	std::vector<std::vector<uint8_t>> imageVector = reader->getImageVector();
	// Read labels into memory for comparison later
	std::vector<uint8_t> labelVector = reader->getLabelVector();

	const int sizeOfInput = 28;
	const int sizeOfFilter = 5;
	const int sizeOfFeature = 24;
	const int sizeOfMaxPool = 12;
	const int sizeOfPoolingWindow = 2;
	const int numberOfFeatures = 8;
	const int numberOfFullyConnectedNodes = 45;
	const int numberOfOutputs = 10;
	const float learningRate = 0.1f;
	const int numberOfEpochs = 10;

	// Initialize the ForwardPropagator
	auto fp = std::make_unique<ForwardPropagator>();
	fp->setSizeOfInput(sizeOfInput);
	fp->setSizeOfFilter(sizeOfFilter);
	fp->setSizeOfFeature(sizeOfFeature);
	fp->setSizeOfMaxPool(sizeOfMaxPool);
	fp->setSizeOfPoolingWindow(sizeOfPoolingWindow);
	fp->setNumberOfFeatures(numberOfFeatures);
	fp->setNumberOfFullyConnectedNodes(numberOfFullyConnectedNodes);
	fp->setNumberOfOutputs(numberOfOutputs);
	fp->randomInitialization();

	auto bp = std::make_unique<BackwardPropagator>();
	bp->setSizeOfInput(sizeOfInput);
	bp->setSizeOfFilter(sizeOfFilter);
	bp->setSizeOfFeature(sizeOfFeature);
	bp->setSizeOfMaxPool(sizeOfMaxPool);
	bp->setSizeOfPoolingWindow(sizeOfPoolingWindow);
	bp->setNumberOfFeatures(numberOfFeatures);
	bp->setNumberOfFullyConnectedNodes(numberOfFullyConnectedNodes);
	bp->setNumberOfOutputs(numberOfOutputs);
	bp->setLearningRate(learningRate);

	int batchSize = 64;
	// for each e
	for (int i = 0; i < numberOfEpochs; i++) {

		auto epoch_start = std::chrono::steady_clock::now();
		std::cout << "Epoch " << i << " started:" << std::endl << std::endl;

		int correct = 0;
		for (int j = 0; j < imageVector.size(); j++) {
			std::vector<float> outputs = fp->forwardPropagation(imageVector[j]);
			std::cout << "Epoch: " << i << "\tDatapoint #: " << j << std::endl;
			std::cout << "Answer: " << unsigned(labelVector[j]) << "\tPrediction: " << maxIndex(outputs) << std::endl;
			for (int z = 0; z < numberOfOutputs; z++)
				std::cout << z << ": " << outputs[z] << " ";
			std::cout << std::endl;
			
			if (unsigned(labelVector[j]) == maxIndex(outputs))
				correct++;

			std::cout << correct << "/" << imageVector.size() << " - " << (float)correct / imageVector.size() * 100 << "% correct" << std::endl;

			bp->setConvolutionalLayerWeights(fp->getConvolutionalLayerWeights());
			bp->setConvolutionalLayerBases(fp->getConvolutionalLayerBases());
			bp->setMaxPoolingLayer(fp->getMaxPoolingLayer());
			bp->setFullyConnectedLayer(fp->getFullyConnectedLayer());
			bp->setFullyConnectedLayerWeights(fp->getFullyConnectedLayerWeights());
			bp->setFullyConnectedLayerBases(fp->getFullyConnectedLayerBases());
			bp->setOutputLayerWeights(fp->getOutputLayerWeights());
			bp->setOutputLayerBases(fp->getOutputLayerBases());

			auto start = std::chrono::steady_clock::now();
			bp->backwardPropagation(imageVector[j], outputs, labelVector[j]);
			auto end = std::chrono::steady_clock::now();

			std::cout << "Back propogation completed in: " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << " seconds" << std::endl << std::endl;

			fp->setConvolutionalLayerWeights(bp->getNewConvolutionalLayerWeights());
			fp->setConvolutionalLayerBases(bp->getNewConvolutionalLayerBases());
			fp->setFullyConnectedLayerWeights(bp->getNewFullyConnectedLayerWeights());
			fp->setFullyConnectedLayerBases(bp->getNewFullyConnectedLayerBases());
			fp->setOutputLayerWeights(bp->getNewOutputLayerWeights());
			fp->setOutputLayerBases(bp->getNewOutputLayerBases());
		}
		// print to weight file
		std::ostringstream fileName;
		fileName << "weight_values_" << i << ".txt";
		std::ofstream weightFile(fileName.str());
		if (weightFile.is_open()) {
			fp->printWeightValues(weightFile, false);
			weightFile.close();
		}

		auto epoch_end = std::chrono::steady_clock::now();
		std::cout << "Epoch " << i << " Finished: " << (float)correct / imageVector.size() * 100 << "% correct" << std::endl;
		std::cout << "Seconds Elapsed: " << std::chrono::duration_cast<std::chrono::seconds>(epoch_end - epoch_start).count() << std::endl << std::endl;
	}

	return 0;
}

void printUsage() {
	using namespace std;

	cout << "Usage:" << endl;
	cout << "\tprog <path-to-training-image-ubyte-file> <path-to-training-label-ubyte-file>" << endl;
}

void printTestImage(std::vector<uint8_t> image) {
	std::ofstream testImageFile("testImage.txt");
	for (int i = 0; i < image.size(); i++) {
		testImageFile << unsigned(image[i]) << " ";
		if ((i + 1) % 28 == 0)
			testImageFile << std::endl;
	}
}

int maxIndex(std::vector<float> values) {
	float max = values[0];
	int maxIndex = 0;

	for (int i = 1; i < values.size(); i++) {
		if (values[i] > max) {
			max = values[i];
			maxIndex = i;
		}
	}

	return maxIndex;
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
