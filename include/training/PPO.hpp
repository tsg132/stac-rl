#pragma once

#include "model/STACFlashModel.hpp"
#include "env/Environment.hpp"
#include "common/Config.hpp"
#include <vector>
#include <memory>

namespace stac::training {

struct TrajectoryBatch {
    std::vector<ObservationTensor> observations;
    std::vector<ActionIndex> actions;
    std::vector<float> rewards;
    std::vector<float> values;
    std::vector<float> log_probs;
    std::vector<bool> dones;
    std::vector<ActionMask> action_masks;
};

class PPOTrainer {
public:
    PPOTrainer(model::STACFlashModel& model, 
               const TrainingConfig& config);
    
    void train(int num_iterations);
    TrajectoryBatch collect_trajectories();
    void update_policy(const TrajectoryBatch& batch);
    
private:
    model::STACFlashModel& model_;
    TrainingConfig config_;
    int iteration_;
};

} // namespace stac::training
