#pragma once

#include "common/Types.hpp"
#include "env/Observation.hpp"
#include <memory>
#include <functional>
#include <algorithm>
#include <iostream>
#include <map>
#include <string>
#include <string_view>

namespace stac::env {

class Environment {

    public:

        Environment();

        virtual ~Environment() = default;

        virtual StepResult reset(const std::string& fen="");

        virtual StepResult step(ActionIndex action_index);

        const Observation& get_observation() const {return observation_;}

        const ActionMask& get_action_mask() const {return action_mask_;}

        bool is_done() const {return done_;}

        Color get_side_to_move() const {return side_to_move_;}

        int get_move_count() const {return move_count_;}

        void set_external_state(const Observation& obs,
                                const ActionMask& mask,
                                Color side_to_move);

        void set_terminal(float reward, GameResult result);

    protected:

        Observation observation_;

        ActionMask action_mask_;

        Color side_to_move_;

        bool done_;

        int move_count_;

        int total_reward_;

        float total_reward_;

        GameResult game_result_;

        std::vector<ActionIndex> action_history_;

        void validate_action(ActionIndex action) const;

        float compute_reward(GameResult result) const;

};

class VectorizedEnvironment {

    public:

        explicit VectorizedEnvironment(int num_envs);

        std::vector<StepResult> reset_all();

        StepResult reset(int env_idx, const std::string& fen="");

        std::vector<StepResult> step(const std::vector<ActionIndex>& actions);

        std::vector<Observation> get_observation() const;

        std::vector<ActionMask> get_actions_masks() const;

        std::vector<bool> get_dones() const;

        int num_envs() const {return envs_.size();}

        int total_steps() const {return total_steps_;}

        float average_episode_length() const;

    private:

        std::vector<std::unique_ptr<Environment>> envs_;

        int total_steps_;

        std::vector<int> episode_length_;

};

class EnvironmentFactory {

    public:

        using CreatorFunc = std::function<std::unique_ptr<Environment>()>;

        static void register_type(const std::string& name, CreatorFunc creator);

        static std::unique_ptr<Environment> create(const std::string& name);

        static std::unique_ptr<VectorizedEnvironment> create_vectorized(const std::string& name, int num_envs);

        static std::vector<std::string> available_types();

    private:

        static std::map<std::string, CreatorFunc>& registry();

};

}