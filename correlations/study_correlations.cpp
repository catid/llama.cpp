#include "compress.hpp"

#include <opencv2/opencv.hpp>

#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

void GenerateHeatmap(CorrelationMatrix& m)
{
    // Calculate basic statistics for e.g. normalization factors

    uint32_t* sorted_values = (uint32_t*)SIMDSafeAllocate(m.WordCount * 4);
    ScopedF sorted_scope([&]() {
        SIMDSafeFree(sorted_values);
    });

    int sorted_count = 0;
    for (int i = 0; i < m.WordCount; ++i) {
        if (m.Data[i]) {
            sorted_values[sorted_count++] = m.Data[i];
        }
    }
    std::sort(sorted_values, sorted_values + sorted_count);

    const int index1 = sorted_count / 100;
    const int index99 = sorted_count * 99 / 100;

    uint32_t value1 = sorted_values[index1];
    uint32_t value99 = sorted_values[index99];
    uint32_t value_max = sorted_values[sorted_count-1];

    cout << "1% value: " << value1 << endl;
    cout << "99% value: " << value99 << endl;
    cout << "max value: " << value_max << endl;

    // Generate heatmap

    const int width = m.MatrixWidth;
    uint8_t* heatmap = SIMDSafeAllocate(width * width);
    ScopedF heatmap_scope([&]() {
        SIMDSafeFree(heatmap);
    });

    for (int i = 0; i < width; ++i) {
        int offset = i * (i + 1) / 2;
        for (int j = 0; j <= i; ++j) {
            int value = m.Data[offset + j];
            float norm_value = (value - value1) / (float)(value99 - value1);
            int heat = norm_value * 255.f;
            if (heat < 0) {
                heat = 0;
            }
            if (heat > 255) {
                heat = 255;
            }

            heatmap[i * width + j] = heat;
            heatmap[j * width + i] = heat;
        }
    }

    // Store as a PNG image

    cv::Mat image(width, width, CV_8UC1, (void*)heatmap);

    std::string filename = "heatmap_block_";
    filename += std::to_string(m.BlockNumber);
    filename += ".png";

    cv::imwrite(filename, image);
}

int main(int argc, char* argv[])
{
    if (argc != 2) {
        cerr << "Expected: study_correlations file1.zstd" << endl;
        return -1;
    }
    const char* m_file = argv[1];

    CorrelationMatrix m;

    if (!m.ReadFile(m_file)) {
        cerr << "Failed to read file: " << m_file << endl;
        return -1;
    }

    cout << "Studying " << m_file << " (Block=" << m.BlockNumber << ", Width=" << m.MatrixWidth << ")" << endl;

    GenerateHeatmap(m);

    return 0;
}
