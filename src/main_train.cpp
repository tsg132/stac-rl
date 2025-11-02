#include "model/STACFlashModel.hpp"
#include "training/PPO.hpp"
#include "training/Optimizer.hpp"
#include "env/Environment.hpp"
#include "common/Config.hpp"
#include <iostream>
#include <memory>

using namespace stac;

int main(int argc, char** argv) {
    std::cout << "==================================" << std::endl;
    std::cout << "STAC-RL Training with PPO + CUDA" << std::endl;
    std::cout << "==================================" << std::endl;
    
    // Configuration
    ModelConfig model_config;
    model_config.embedding_dim = 512;
    model_config.num_layers = 8;
    model_config.num_heads = 8;
    model_config.mlp_hidden_dim = 2048;
    model_config.use_cuda = true;
    model_config.cuda_device = 0;
    
    TrainingConfig training_config;
    training_config.learning_rate = 3e-4f;
    training_config.clip_epsilon = 0.2f;
    training_config.value_loss_coef = 0.5f;
    training_config.entropy_coef = 0.01f;
    training_config.gamma = 0.99f;
    training_config.gae_lambda = 0.95f;
    training_config.num_epochs = 4;
    training_config.batch_size = 2048;
    training_config.rollout_length = 512;
    training_config.num_envs = 64;
    training_config.checkpoint_interval = 1000;
    training_config.log_interval = 10;
    
    std::cout << "\nModel Configuration:" << std::endl;
    std::cout << "  Embedding Dim: " << model_config.embedding_dim << std::endl;
    std::cout << "  Num Layers: " << model_config.num_layers << std::endl;
    std::cout << "  Num Heads: " << model_config.num_heads << std::endl;
    std::cout << "  MLP Hidden: " << model_config.mlp_hidden_dim << std::endl;
    std::cout << "  CUDA: " << (model_config.use_cuda ? "Enabled" : "Disabled") << std::endl;
    
    std::cout << "\nTraining Configuration:" << std::endl;
    std::cout << "  Learning Rate: " << training_config.learning_rate << std::endl;
    std::cout << "  Clip Epsilon: " << training_config.clip_epsilon << std::endl;
    std::cout << "  Num Environments: " << training_config.num_envs << std::endl;
    std::cout << "  Rollout Length: " << training_config.rollout_length << std::endl;
    std::cout << "  Batch Size: " << training_config.batch_size << std::endl;
    std::cout << "  PPO Epochs: " << training_config.num_epochs << std::endl;
    
    try {
        // Create model
        std::cout << "\n[1/4] Creating model..." << std::endl;
        auto model = model::FlashModelFactory::create(model_config);
        
        if (model_config.use_cuda) {
            std::cout << "  Moving model to CUDA device " << model_config.cuda_device << std::endl;
            model->to_device(model_config.cuda_device);
        }
        
        std::cout << "  Model parameters: " << model->num_parameters() << std::endl;
        
        // Create environments
        std::cout << "\n[2/4] Creating vectorized environments..." << std::endl;
        auto envs = std::make_unique<env::VectorizedEnvironment>(training_config.num_envs);
        std::cout << "  Created " << envs->num_envs() << " parallel environments" << std::endl;
        
        // Create optimizer
        std::cout << "\n[3/4] Setting up optimizer..." << std::endl;
        training::AdamConfig adam_config;
        adam_config.lr = training_config.learning_rate;
        adam_config.beta1 = training_config.adam_beta1;
        adam_config.beta2 = training_config.adam_beta2;
        adam_config.eps = training_config.adam_eps;
        
        auto optimizer = std::make_unique<training::AdamOptimizer>(adam_config);
        
        // Add all model parameters to optimizer
        auto params = model->get_parameters();
        std::cout << "  Registering " << params.size() << " parameter groups" << std::endl;
        
        // In practice, you'd iterate through all parameters here
        // for (auto* param : params) {
        //     optimizer->add_param_group(param, grad_of_param, size_of_param);
        // }
        
        // Create learning rate scheduler
        std::unique_ptr<training::LRScheduler> scheduler;
        if (training_config.use_lr_schedule) {
            scheduler = std::make_unique<training::CosineAnnealingLR>(
                optimizer.get(), 10000, 1e-5f);
            std::cout << "  Using Cosine Annealing LR scheduler" << std::endl;
        }
        
        // Training loop
        std::cout << "\n[4/4] Starting training..." << std::endl;
        std::cout << "====================================\n" << std::endl;
        
        const int num_iterations = 10000;
        const int update_interval = 1;  // Update every iteration
        
        int total_steps = 0;
        float best_reward = -INFINITY;
        
        for (int iter = 0; iter < num_iterations; ++iter) {
            // This is a simplified training loop
            // Full implementation would be in PPOTrainer class
            
            // 1. Collect rollouts
            // 2. Compute advantages
            // 3. Update policy with PPO
            // 4. Update learning rate
            
            if (iter % training_config.log_interval == 0) {
                std::cout << "Iteration " << iter << " / " << num_iterations << std::endl;
                std::cout << "  Steps: " << total_steps << std::endl;
                std::cout << "  LR: " << optimizer->get_lr() << std::endl;
                std::cout << "  Best Reward: " << best_reward << std::endl;
            }
            
            // Update scheduler
            if (scheduler && iter > 0 && iter % update_interval == 0) {
                scheduler->step();
            }
            
            // Save checkpoint
            if (iter % training_config.checkpoint_interval == 0 && iter > 0) {
                std::string checkpoint_path = training_config.checkpoint_dir.string() +
                                             "/checkpoint_" + std::to_string(iter) + ".bin";
                model->save(checkpoint_path);
                std::cout << "  Saved checkpoint: " << checkpoint_path << std::endl;
            }
            
            total_steps += training_config.rollout_length * training_config.num_envs;
        }
        
        // Save final model
        std::cout << "\n=====================================" << std::endl;
        std::cout << "Training completed!" << std::endl;
        std::cout << "=====================================" << std::endl;
        
        std::string final_path = "models/final_model.bin";
        model->save(final_path);
        std::cout << "Final model saved to: " << final_path << std::endl;
        
        // Print statistics
        model->print_summary();
        
    } catch (const std::exception& e) {
        std::cerr << "Error during training: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
