    def compute_reward(fire: np.array, agent_pos: Tuple[int, int]) -> Tuple[float, Dict[str, float]]:
        # Setting up reward components
        distance_reward = 0.0
        fire_reward = 0.0
    
        # Temperature parameters for transformation
        distance_temperature = 5.0
        fire_temperature = 12.5
    
        # Getting agent position
        agent_x, agent_y = agent_pos
    
        # Calculating Euclidean distance from each fire and Sum it up
        total_distance = 0.0
        for i in range(fire.shape[0]):
            for j in range(fire.shape[1]):
                # If a cell is on fire
                if fire[i, j] == 1:
                    total_distance += np.sqrt((agent_x - i) ** 2 + (agent_y - j) ** 2)
        
        # Inversely proportional to the distance from the fire
        distance_reward = 1 / total_distance if total_distance > 0 else 0
    
        # Count of fires in the environment
        num_fires = np.sum(fire)
    
        # Inversely proportional to the number of fires
        fire_reward = 1 / num_fires if num_fires > 0 else 0
    
        # Normalize rewards with temperature parameter using exponential function
        distance_reward = np.exp(distance_reward / distance_temperature)
        fire_reward = np.exp(fire_reward / fire_temperature)
    
        # Weighting the different reward components
        final_reward = 0.7 * distance_reward + 0.3 * fire_reward
    
        # Returning final reward and individual reward components
        return final_reward, {'distance_reward': distance_reward, 'fire_reward': fire_reward}
