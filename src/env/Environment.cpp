/**
 * Chess Environment Implementation
 */

#include "env/Environment.hpp"
#include <stdexcept>

namespace stac::env {

Environment::Environment() 
    : side_to_move_(Color::WHITE)
    , done_(false)
    , move_count_(0)
    , total_reward_(0.0f)
    , game_result_(GameResult::ONGOING) {
}

StepResult Environment::reset(const std::string& fen) {
    // Reset environment to starting position
    done_ = false;
    move_count_ = 0;
    total_reward_ = 0.0f;
    game_result_ = GameResult::ONGOING;
    side_to_move_ = Color::WHITE;
    action_history_.clear();
    
    // TODO: Parse FEN and set observation/mask
    std::fill(observation_.begin(), observation_.end(), 0.0f);
    std::fill(action_mask_.begin(), action_mask_.end(), 0);
    
    return StepResult{observation_, action_mask_, 0.0f, false, game_result_};
}

StepResult Environment::step(ActionIndex action_index) {
    validate_action(action_index);
    
    // TODO: Apply action and update state
    action_history_.push_back(action_index);
    move_count_++;
    
    // Placeholder reward
    float reward = 0.0f;
    
    // Check if game is done
    // TODO: Implement actual game over check
    
    return StepResult{observation_, action_mask_, reward, done_, game_result_};
}

void Environment::set_external_state(const Observation& obs,
                                     const ActionMask& mask,
                                     Color side_to_move) {
    observation_ = obs;
    action_mask_ = mask;
    side_to_move_ = side_to_move;
}

void Environment::set_terminal(float reward, GameResult result) {
    done_ = true;
    total_reward_ = reward;
    game_result_ = result;
}

void Environment::validate_action(ActionIndex action) const {
    if (action < 0 || action >= static_cast<ActionIndex>(action_mask_.size())) {
        throw std::out_of_range("Action index out of range: " + std::to_string(action));
    }
    
    if (action_mask_[action] == 0) {
        throw std::invalid_argument("Illegal action attempted: " + std::to_string(action));
    }
}

float Environment::compute_reward(GameResult result) const {
    switch (result) {
        case GameResult::WHITE_WIN:
            return side_to_move_ == Color::WHITE ? 1.0f : -1.0f;
        case GameResult::BLACK_WIN:
            return side_to_move_ == Color::BLACK ? 1.0f : -1.0f;
        case GameResult::DRAW:
            return 0.0f;
        default:
            return 0.0f;
    }
}

// ============================================================================
// Vectorized Environment
// ============================================================================

VectorizedEnvironment::VectorizedEnvironment(int num_envs) 
    : total_steps_(0) {
    
    envs_.reserve(num_envs);
    episode_length_.resize(num_envs, 0);
    
    for (int i = 0; i < num_envs; ++i) {
        envs_.push_back(std::make_unique<Environment>());
    }
}

std::vector<StepResult> VectorizedEnvironment::reset_all() {
    std::vector<StepResult> results;
    results.reserve(envs_.size());
    
    for (auto& env : envs_) {
        results.push_back(env->reset());
    }
    
    return results;
}

StepResult VectorizedEnvironment::reset(int env_idx, const std::string& fen) {
    if (env_idx < 0 || env_idx >= static_cast<int>(envs_.size())) {
        throw std::out_of_range("Environment index out of range");
    }
    
    episode_length_[env_idx] = 0;
    return envs_[env_idx]->reset(fen);
}

std::vector<StepResult> VectorizedEnvironment::step(const std::vector<ActionIndex>& actions) {
    if (actions.size() != envs_.size()) {
        throw std::invalid_argument("Number of actions must match number of environments");
    }
    
    std::vector<StepResult> results;
    results.reserve(envs_.size());
    
    for (size_t i = 0; i < envs_.size(); ++i) {
        auto result = envs_[i]->step(actions[i]);
        results.push_back(result);
        
        episode_length_[i]++;
        total_steps_++;
        
        if (result.done) {
            episode_length_[i] = 0;
        }
    }
    
    return results;
}

std::vector<Observation> VectorizedEnvironment::get_observation() const {
    std::vector<Observation> observations;
    observations.reserve(envs_.size());
    
    for (const auto& env : envs_) {
        observations.push_back(env->get_observation());
    }
    
    return observations;
}

std::vector<ActionMask> VectorizedEnvironment::get_actions_masks() const {
    std::vector<ActionMask> masks;
    masks.reserve(envs_.size());
    
    for (const auto& env : envs_) {
        masks.push_back(env->get_action_mask());
    }
    
    return masks;
}

std::vector<bool> VectorizedEnvironment::get_dones() const {
    std::vector<bool> dones;
    dones.reserve(envs_.size());
    
    for (const auto& env : envs_) {
        dones.push_back(env->is_done());
    }
    
    return dones;
}

float VectorizedEnvironment::average_episode_length() const {
    if (episode_length_.empty()) {
        return 0.0f;
    }
    
    int sum = 0;
    for (int len : episode_length_) {
        sum += len;
    }
    
    return static_cast<float>(sum) / episode_length_.size();
}

// ============================================================================
// Environment Factory
// ============================================================================

std::map<std::string, EnvironmentFactory::CreatorFunc>& EnvironmentFactory::registry() {
    static std::map<std::string, CreatorFunc> registry_;
    return registry_;
}

void EnvironmentFactory::register_type(const std::string& name, CreatorFunc creator) {
    registry()[name] = creator;
}

std::unique_ptr<Environment> EnvironmentFactory::create(const std::string& name) {
    auto& reg = registry();
    auto it = reg.find(name);
    
    if (it == reg.end()) {
        throw std::runtime_error("Unknown environment type: " + name);
    }
    
    return it->second();
}

std::unique_ptr<VectorizedEnvironment> EnvironmentFactory::create_vectorized(
    const std::string& name, 
    int num_envs) {
    
    // For now, just create a vectorized environment with default chess envs
    return std::make_unique<VectorizedEnvironment>(num_envs);
}

std::vector<std::string> EnvironmentFactory::available_types() {
    std::vector<std::string> types;
    for (const auto& pair : registry()) {
        types.push_back(pair.first);
    }
    return types;
}

} // namespace stac::env
