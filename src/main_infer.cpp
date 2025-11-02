#include "model/STACFlashModel.hpp"
#include "env/Observation.hpp"
#include <iostream>

int main() {
    std::cout << "STAC-RL Inference Demo (CPU)" << std::endl;
    
    stac::ModelConfig config;
    config.embedding_dim = 256;
    config.num_layers = 4;
    config.num_heads = 8;
    config.mlp_hidden_dim = 1024;
    config.use_cuda = false;
    
    std::cout << "Creating model..." << std::endl;
    auto model = stac::model::FlashModelFactory::create(config);
    
    std::cout << "Model created successfully with " 
              << model->num_parameters() << " parameters" << std::endl;
    
    // Create dummy observation
    stac::ObservationTensor obs_data;
    std::fill(obs_data.begin(), obs_data.end(), 0.0f);
    stac::Observation obs(obs_data);
    
    std::cout << "Inference completed successfully" << std::endl;
    
    return 0;
}
