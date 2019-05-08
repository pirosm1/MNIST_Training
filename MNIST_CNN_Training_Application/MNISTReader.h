#pragma once
#include <vector>
#include <iostream>
#include <fstream>
#include <string>

class MNISTReader {
private:
	inline uint32_t readHeader(const std::unique_ptr<char[]>& buffer, size_t position) {
		auto header = reinterpret_cast<uint32_t*>(buffer.get());

		auto value = *(header + position);

		return (value << 24) | ((value << 8) & 0x00FF0000) | ((value >> 8) & 0X0000FF00) | (value >> 24);
	}

	inline std::unique_ptr<char[]> readMNISTFile(const std::string& path, uint32_t key) {
		std::ifstream file;
		file.open(path, std::ios::in | std::ios::binary | std::ios::ate);

		std::cout << "Attempting to open file: " + path << std::endl;

		if (!file) {
			std::cout << "Error opening file" << std::endl;
			return {};
		}

		std::cout << "Reading data..." << std::endl;

		auto size = (size_t)file.tellg();
		auto buffer = std::make_unique<char[]>(size);
		//Read the entire file at once
		file.seekg(0, std::ios::beg);
		file.read(buffer.get(), size);
		file.close();

		auto magic = readHeader(buffer, 0);

		if (magic != key) {
			std::cout << "Invalid magic number, probably not a MNIST file" << std::endl;
			return {};
		}

		auto count = readHeader(buffer, 1);

		if (magic == 0x803) {
			auto rows = readHeader(buffer, 2);
			auto columns = readHeader(buffer, 3);

			if (size < count * rows * columns + 16) {
				std::cout << "The file is not large enough to hold all the data, probably corrupted" << std::endl;
				return {};
			}
		}
		else if (magic == 0x801) {
			if (size < count + 8) {
				std::cout << "The file is not large enough to hold all the data, probably corrupted" << std::endl;
				return {};
			}
		}

		return buffer;
	}
public:
	MNISTReader();
	~MNISTReader();

	std::vector<std::vector<uint8_t>> getImageVector(std::string fileName);

	std::vector<uint8_t> getLabelVector(std::string fileName);
};