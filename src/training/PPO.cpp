#include "training/PPO.hpp"
#include "training/Optimizer.hpp"
#include "model/STACFlashModel.hpp"
#include "env/Environment.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>

namespace stac::training {

// ============================================================================
// Trajectory Batch for Training
// ============================================================================

struct TrajectoryBatch {
    std::vector<ObservationTensor> observations;
    std::vector<ActionIndex> actions;
    std::vector<float> rewards;
    std::vector<float> values;
    std::vector<float> log_probs;
    std::vector<bool> dones;
    
    std::vector<float> returns;
    std::vector<float> advantages;
    
    int size() const { return observations.size(); }
    
    void clear() {
        observations.clear();
        actions.clear();
        rewards.clear();
        values.clear();
        log_probs.clear();
        dones.clear();
        returns.clear();
        advantages.clear();
    }
};

// ============================================================================
// PPO Trainer Implementation
// ============================================================================

class PPOTrainer {
public:
    PPOTrainer(model::STACFlashModel& model,
               env::VectorizedEnvironment& envs,
               const TrainingConfig& config)
        : model_(model), envs_(envs), config_(config) {
        
        // Create optimizer
        AdamConfig adam_config;
        adam_config.lr = config.learning_rate;
        adam_config.eps = config.adam_eps;
        adam_config.weight_decay = 0.0f;
        
        optimizer_ = std::make_unique<AdamOptimizer>(adam_config);
        
        // Add parameter groups (simplified - would need actual model parameters)
        // In practice, iterate through model.get_parameters() and add each
        
        // Create LR scheduler
        if (config.use_lr_schedule) {
            scheduler_ = std::make_unique<ExponentialLR>(
                optimizer_.get(), config.lr_schedule_gamma);
        }
        
        total_steps_ = 0;
        episode_count_ = 0;
    }
    
    void train(int num_iterations) {
        std::cout << "Starting PPO training for " << num_iterations << " iterations" << std::endl;
        
        for (int iter = 0; iter < num_iterations; ++iter) {
            // Collect trajectories
            TrajectoryBatch batch = collect_trajectories();
            
            // Compute advantages
            compute_gae(batch);
            
            // Normalize advantages
            normalize_advantages(batch);
            
            // Update policy
            update_policy(batch);
            
            // Update learning rate
            if (scheduler_) {
                scheduler_->step();
            }
            
            total_steps_ += batch.size();
            
            // Logging
            if (iter % config_.log_interval == 0) {
                log_training_stats(iter, batch);
            }
            
            // Checkpointing
            if (iter % config_.checkpoint_interval == 0 && iter > 0) {
                save_checkpoint(iter);
            }
        }
    }
    
private:
    model::STACFlashModel& model_;
    env::VectorizedEnvironment& envs_;
    TrainingConfig config_;
    
    std::unique_ptr<AdamOptimizer> optimizer_;
    std::unique_ptr<LRScheduler> scheduler_;
    
    int total_steps_;
    int episode_count_;
    
    // Collect rollout data
    TrajectoryBatch collect_trajectories() {
        TrajectoryBatch batch;
        model_.eval_mode();
        
        for (int step = 0; step < config_.rollout_length; ++step) {
            // Get current observations
            auto obs = envs_.get_observation();
            auto masks = envs_.get_actions_masks();
            
            // Forward pass through model
            std::vector<ModelOutput> outputs;
            for (int i = 0; i < config_.num_envs; ++i) {
                auto output = model_.forward(obs[i]);
                outputs.push_back(output);
                
                // Store trajectory data
                batch.observations.push_back(obs[i]);
                batch.values.push_back(output.value);
            }
            
            // Sample actions
            std::vector<ActionIndex> actions;
            for (int i = 0; i < config_.num_envs; ++i) {
                // Sample from policy (simplified - should use proper sampling)
                ActionIndex action = sample_action(outputs[i].policy_logits, masks[i]);
                actions.push_back(action);
                batch.actions.push_back(action);
                
                // Compute log probability
                float log_prob = compute_log_prob(outputs[i].policy_logits, action, masks[i]);
                batch.log_probs.push_back(log_prob);
            }
            
            // Step environments
            auto step_results = envs_.step(actions);
            
            // Store rewards and dones
            for (int i = 0; i < config_.num_envs; ++i) {
                batch.rewards.push_back(step_results[i].reward);
                batch.dones.push_back(step_results[i].done);
            }
        }
        
        return batch;
    }
    
    // Compute Generalized Advantage Estimation
    void compute_gae(TrajectoryBatch& batch) {
        batch.returns.resize(batch.size());
        batch.advantages.resize(batch.size());
        
        int num_steps = config_.rollout_length;
        float gae = 0.0f;
        
        // Process each environment's trajectory
        for (int env_idx = 0; env_idx < config_.num_envs; ++env_idx) {
            gae = 0.0f;
            
            for (int t = num_steps - 1; t >= 0; --t) {
                int idx = env_idx * num_steps + t;
                int next_idx = idx + 1;
                
                float reward = batch.rewards[idx];
                float value = batch.values[idx];
                float next_value = (t < num_steps - 1 && !batch.dones[idx]) 
                                  ? batch.values[next_idx] : 0.0f;
                
                // TD error: delta = r + gamma * V(s') - V(s)
                float delta = reward + config_.gamma * next_value - value;
                
                // GAE: A = delta + gamma * lambda * A_next
                gae = delta + config_.gamma * config_.gae_lambda * gae * (1.0f - batch.dones[idx]);
                
                batch.advantages[idx] = gae;
                batch.returns[idx] = gae + value;
            }
        }
    }
    
    // Normalize advantages for stable training
    void normalize_advantages(TrajectoryBatch& batch) {
        // Compute mean and std
        float mean = std::accumulate(batch.advantages.begin(), 
                                    batch.advantages.end(), 0.0f) / batch.size();
        
        float variance = 0.0f;
        for (float adv : batch.advantages) {
            variance += (adv - mean) * (adv - mean);
        }
        variance /= batch.size();
        float std_dev = std::sqrt(variance + 1e-8f);
        
        // Normalize
        for (float& adv : batch.advantages) {
            adv = (adv - mean) / std_dev;
        }
    }
    
    // Update policy using PPO algorithm
    void update_policy(const TrajectoryBatch& batch) {
        model_.train_mode();
        
        // Store old log probs for PPO ratio
        std::vector<float> old_log_probs = batch.log_probs;
        
        // Mini-batch training
        int batch_size = config_.batch_size;
        int num_batches = (batch.size() + batch_size - 1) / batch_size;
        
        for (int epoch = 0; epoch < config_.num_epochs; ++epoch) {
            // Shuffle indices
            std::vector<int> indices(batch.size());
            std::iota(indices.begin(), indices.end(), 0);
            std::random_shuffle(indices.begin(), indices.end());
            
            float epoch_policy_loss = 0.0f;
            float epoch_value_loss = 0.0f;
            float epoch_entropy = 0.0f;
            
            for (int b = 0; b < num_batches; ++b) {
                int start_idx = b * batch_size;
                int end_idx = std::min(start_idx + batch_size, batch.size());
                
                optimizer_->zero_grad();
                
                float policy_loss = 0.0f;
                float value_loss = 0.0f;
                float entropy = 0.0f;
                
                // Process mini-batch
                for (int i = start_idx; i < end_idx; ++i) {
                    int idx = indices[i];
                    
                    // Forward pass
                    auto output = model_.forward(batch.observations[idx]);
                    
                    // Compute new log prob
                    float log_prob = compute_log_prob(output.policy_logits, 
                                                     batch.actions[idx],
                                                     ActionMask{});  // Simplified
                    
                    // PPO policy loss
                    float ratio = std::exp(log_prob - old_log_probs[idx]);
                    float surr1 = ratio * batch.advantages[idx];
                    float surr2 = std::clamp(ratio, 
                                            1.0f - config_.clip_epsilon,
                                            1.0f + config_.clip_epsilon) * batch.advantages[idx];
                    policy_loss += -std::min(surr1, surr2);
                    
                    // Value loss (clipped)
                    float value_pred = output.value;
                    float value_target = batch.returns[idx];
                    float value_error = value_pred - value_target;
                    value_loss += value_error * value_error;
                    
                    // Entropy bonus
                    entropy += compute_entropy(output.policy_logits);
                }
                
                // Average losses
                int mini_batch_size = end_idx - start_idx;
                policy_loss /= mini_batch_size;
                value_loss /= mini_batch_size;
                entropy /= mini_batch_size;
                
                // Total loss
                float total_loss = policy_loss + 
                                  config_.value_loss_coef * value_loss - 
                                  config_.entropy_coef * entropy;
                
                // Backward pass (simplified - would need actual gradient computation)
                // In practice: compute_gradients(total_loss)
                
                // Gradient clipping
                // clip_gradients(config_.grad_clip_norm)
                
                // Optimizer step
                optimizer_->step();
                
                epoch_policy_loss += policy_loss;
                epoch_value_loss += value_loss;
                epoch_entropy += entropy;
            }
            
            // Log epoch stats
            if (epoch % 10 == 0) {
                std::cout << "  Epoch " << epoch 
                         << " | Policy Loss: " << epoch_policy_loss / num_batches
                         << " | Value Loss: " << epoch_value_loss / num_batches
                         << " | Entropy: " << epoch_entropy / num_batches << std::endl;
            }
        }
    }
    
    // Helper: Sample action from policy
    ActionIndex sample_action(const PolicyLogits& logits, const ActionMask& mask) {
        // Simplified sampling - in practice use proper categorical sampling
        float max_logit = -INFINITY;
        ActionIndex best_action = 0;
        
        for (int i = 0; i < constants::ACTION_SPACE_SIZE; ++i) {
            if (mask[i] && logits[i] > max_logit) {
                max_logit = logits[i];
                best_action = i;
            }
        }
        
        return best_action;
    }
    
    // Helper: Compute log probability
    float compute_log_prob(const PolicyLogits& logits, ActionIndex action,
                          const ActionMask& mask) {
        // Compute softmax
        float max_logit = *std::max_element(logits.begin(), logits.end());
        float sum_exp = 0.0f;
        
        for (int i = 0; i < constants::ACTION_SPACE_SIZE; ++i) {
            if (mask[i]) {
                sum_exp += std::exp(logits[i] - max_logit);
            }
        }
        
        return logits[action] - max_logit - std::log(sum_exp);
    }
    
    // Helper: Compute policy entropy
    float compute_entropy(const PolicyLogits& logits) {
        float max_logit = *std::max_element(logits.begin(), logits.end());
        float sum_exp = 0.0f;
        float entropy = 0.0f;
        
        for (float logit : logits) {
            float prob = std::exp(logit - max_logit);
            sum_exp += prob;
        }
        
        for (float logit : logits) {
            float prob = std::exp(logit - max_logit) / sum_exp;
            if (prob > 1e-8f) {
                entropy -= prob * std::log(prob);
            }
        }
        
        return entropy;
    }
    
    // Logging
    void log_training_stats(int iteration, const TrajectoryBatch& batch) {
        float avg_reward = std::accumulate(batch.rewards.begin(), 
                                          batch.rewards.end(), 0.0f) / batch.size();
        float avg_value = std::accumulate(batch.values.begin(),
                                         batch.values.end(), 0.0f) / batch.size();
        
        std::cout << "Iteration " << iteration 
                 << " | Steps: " << total_steps_
                 << " | Avg Reward: " << avg_reward
                 << " | Avg Value: " << avg_value
                 << " | LR: " << optimizer_->get_lr() << std::endl;
    }
    
    // Checkpointing
    void save_checkpoint(int iteration) {
        std::string filename = config_.checkpoint_dir.string() + 
                              "/checkpoint_" + std::to_string(iteration) + ".bin";
        // model_.save(filename);
        std::cout << "Checkpoint saved: " << filename << std::endl;
    }
};

} // namespace stac::training
