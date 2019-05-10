#include "pch.h"
#include "MNISTReader.h"

MNISTReader::MNISTReader() {}

std::vector<std::vector<uint8_t>> MNISTReader::getImageVector(std::string fileName) {

	auto buffer = readMNISTFile(fileName, 0x803);

	if (!buffer) {
		return {};
	}
	auto count   = readHeader(buffer, 1);
	auto rows    = readHeader(buffer, 2);
	auto columns = readHeader(buffer, 3);

	// Skip header and then cast to unsigned char just in case platform needs it
	auto image_buffer = reinterpret_cast<unsigned char*>(buffer.get() + 16);

	std::vector<std::vector<uint8_t>> images;
	images.reserve(count);

	for (size_t i = 0; i < count; ++i) {
		images.emplace_back(rows * columns);

		for (size_t j = 0; j < rows * columns; ++j) {
			auto pixel   = *image_buffer++;
			images[i][j] = static_cast<uint8_t>(pixel);
		}
	}

	std::cout << "Successfully created imageVector" << std::endl << std::endl;

	return images;
}

std::vector<uint8_t> MNISTReader::getLabelVector(std::string fileName) {
	auto buffer = readMNISTFile(fileName, 0x801);

	if (!buffer) {
		return {};
	}

	auto count = readHeader(buffer, 1);

	// Skip header and then cast to unsigned char just in case platform needs it
	auto labelBuffer = reinterpret_cast<unsigned char*>(buffer.get() + 8);

	std::vector<uint8_t> labels(count);

    for (size_t i = 0; i < count; ++i) {
        auto label = *labelBuffer++;
        labels[i]  = static_cast<uint8_t>(label);
    }

	std::cout << "Successfully created labelVector" << std::endl << std::endl;

	return labels;
}

MNISTReader::~MNISTReader() {}
