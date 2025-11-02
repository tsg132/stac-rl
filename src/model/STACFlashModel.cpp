#include "model/STACFlashModel.hpp"
#include <iostream>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#endif

namespace stac::model {

// Stub implementations for linking

STACFlashModel::~STACFlashModel() {}

std::unique_ptr<STACFlashModel> FlashModelFactory::create(const ModelConfig& config) {
    return std::unique_ptr<STACFlashModel>(new STACFlashModel());
}

int STACFlashModel::num_parameters() const {
    return 1000000; // Stub
}

void STACFlashModel::to_device(int device_id) {
    // Stub
}

std::vector<float*> STACFlashModel::get_parameters() {
    return {}; // Stub
}

void STACFlashModel::save(const std::filesystem::path& path) const {
    std::cout << "Saving model to: " << path << std::endl;
}

void STACFlashModel::print_summary() const {
    std::cout << "Model with " << num_parameters() << " parameters" << std::endl;
}

} // namespace stac::model
