#include "compress.hpp"

#include <opencv2/opencv.hpp>

#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

static void GenerateHeatmap(CorrelationMatrix& m)
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

    const int width = m.MatrixWidth;

    // Histogram

    const int hist_w = 8000; int hist_h = 4000;
    uint32_t* hdata = (uint32_t*)SIMDSafeAllocate(width * 4);
    uint32_t* histogram = (uint32_t*)SIMDSafeAllocate(hist_w * 4);
    uint8_t* histogram_image = SIMDSafeAllocate(hist_w * hist_h);
    ScopedF histogram_scope([&]() {
        SIMDSafeFree(hdata);
        SIMDSafeFree(histogram);
        SIMDSafeFree(histogram_image);
    });

    for (int i = 0; i < width; ++i) {
        const int diagonal_j = i * (i + 1) / 2;
        uint32_t value = m.Data[diagonal_j];
        hdata[i] = value;
    }
    std::sort(hdata, hdata + width);

    for (int i = 0; i < hist_w; ++i) {
        histogram[i] = 0;
    }

    for (int i = 0; i < width; ++i) {
        int bin = i * hist_w / width;
        if (bin < 0) {
            bin = 0;
        } else if (bin >= width) {
            bin = width - 1;
        }

        // Note this has an aliasing artifact that will cause this to jump up and down based on variance.
        // I kind of like this because it visually demonstrates variance in the data without additional effort.
        histogram[bin] += hdata[i];
    }

    uint32_t hist_max = histogram[0];
    for (int i = 1; i < hist_w; ++i) {
        if (hist_max < histogram[i]) {
            hist_max = histogram[i];
        }
    }

    for (int i = 0; i < hist_w; ++i) {
        int height = (int)(histogram[i] / (double)hist_max * hist_h);
        if (height < 0) {
            height = 0;
        } else if (height >= hist_h) {
            height = hist_h;
        }

        for (int j = 0; j < hist_h; ++j) {
            histogram_image[i + (hist_h - 1 - j) * hist_w] = j > height ? 0 : 255;
        }
    }

    // Write histogram

    cv::Mat hist_image(hist_h, hist_w, CV_8UC1, (void*)histogram_image);

    std::string hist_filename = "histogram_block_";
    hist_filename += std::to_string(m.BlockNumber);
    hist_filename += ".png";

    cv::imwrite(hist_filename, hist_image);

    // Correlation matrix calculation

    // Note: The matrix is symmetric, so row stats are the same as column stats
    std::vector<double> StdDevs(width), Means(width);

    double norm_factor = 1.0 / value_max;

    for (int i = 0; i < width; ++i) {
        // Calculate mean of row:

        uint64_t sum_values = 0;

        // For each row:
        int i_offset = i * (i + 1) / 2;
        for (int j = 0; j <= i; ++j) {
            sum_values += m.Data[i_offset + j];
        }
        for (int j = i+1; j < width; ++j) {
            int j_offset = j * (j + 1) / 2;
            sum_values += m.Data[j_offset + i];
        }

        double mean = sum_values / (double)width * norm_factor;
        Means[i] = mean;

        // Calculate standard deviation of row:

        double sum_sd = 0.0;

        // For each row:
        for (int j = 0; j <= i; ++j) {
            uint32_t value = m.Data[i_offset + j];
            double diff = value * norm_factor - mean;
            sum_sd += diff * diff;
        }
        for (int j = i+1; j < width; ++j) {
            int j_offset = j * (j + 1) / 2;
            uint32_t value = m.Data[j_offset + i];
            double diff = value * norm_factor - mean;
            sum_sd += diff * diff;
        }

        StdDevs[i] = std::sqrt(sum_sd);
    }


    // Generate heatmap

    uint8_t* heatmap = SIMDSafeAllocate(width * width * 3);
    ScopedF heatmap_scope([&]() {
        SIMDSafeFree(heatmap);
    });

    for (int i = 0; i < width; ++i)
    {
        int offset = i * (i + 1) / 2;

        for (int j = 0; j <= i; ++j)
        {
            double norm_value = m.Data[offset + j] * norm_factor;
            double cov_ij = (norm_value - Means[i]) * (norm_value - Means[j]);
            double r = cov_ij / (StdDevs[i] * StdDevs[j]);

            int heat = r * 255.0;

            int offset_ij = i * width + j;
            int offset_ji = j * width + i;
            if (heat >= 0) {
                if (heat > 255) {
                    heat = 255;
                }
                heatmap[offset_ij * 3 + 0] = (uint8_t)heat;
                heatmap[offset_ij * 3 + 1] = (uint8_t)heat;
                heatmap[offset_ij * 3 + 2] = (uint8_t)heat;
                heatmap[offset_ji * 3 + 0] = (uint8_t)heat;
                heatmap[offset_ji * 3 + 1] = (uint8_t)heat;
                heatmap[offset_ji * 3 + 2] = (uint8_t)heat;
            }
            else if (heat < 0) {
                heat = -heat;
                if (heat > 255) {
                    heat = 255;
                }
                heatmap[offset_ij * 3 + 0] = 0;
                heatmap[offset_ij * 3 + 1] = (uint8_t)heat;
                heatmap[offset_ij * 3 + 2] = 0;
                heatmap[offset_ji * 3 + 0] = 0;
                heatmap[offset_ji * 3 + 1] = (uint8_t)heat;
                heatmap[offset_ji * 3 + 2] = 0;
            }
        }
    }

    // Store as a PNG image

    cv::Mat heatmap_image(width, width, CV_8UC3, (void*)heatmap);

    std::string heatmap_filename = "heatmap_block_";
    heatmap_filename += std::to_string(m.BlockNumber);
    heatmap_filename += ".png";

    cv::imwrite(heatmap_filename, heatmap_image);
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
