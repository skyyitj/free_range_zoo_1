    def compute_reward(observation: WildfireObservation, action: np.ndarray) -> Tuple[np.array, Dict[str, np.array]]:
    
        # constants for reward adjustment
        SUCCESS_REWARD = 100.0
        FAILURE_REWARD = -100.0
        SUPPRESSANT_USAGE_PENALTY = -0.5
        FIRE_INTENSITY_PENALTY = -1.0
        ATTACK_PENALTY = -0.1
    
        reward = 0.0
        reward_components = {}
    
        # calculate fires left
        fires_left = np.sum(observation.fires * observation.intensity)
    
        if fires_left > 0:
            # Still some fires left
            # calculate reward based on the reduction of fire and the usage of the suppressants
            suppressants_usage_reward = SUPPRESSANT_USAGE_PENALTY * np.sum(action)
            fire_intensity_reward = FIRE_INTENSITY_PENALTY * np.sum(observation.fires * observation.intensity)
            attack_count_reward = ATTACK_PENALTY * np.sum(observation.attack_counts)
    
            reward = suppressants_usage_reward + fire_intensity_reward + attack_count_reward
            reward_components = {"suppressants_usage_reward": suppressants_usage_reward, 
                                 "fire_intensity_reward": fire_intensity_reward,
                                 "attack_count_reward": attack_count_reward}
    
        else:
            # Fires extinguished
            reward = SUCCESS_REWARD
            reward_components = {"success_reward": SUCCESS_REWARD}
    
        return reward, reward_components
