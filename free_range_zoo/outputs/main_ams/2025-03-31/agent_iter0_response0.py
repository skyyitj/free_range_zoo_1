# Auto-generated Agent Class
from typing import Tuple, Dict
import torch
from torch import Tensor

class GenerateAgent:
    """Agent that prioritizes attacking the strongest fire within range while managing resources efficiently."""

    def __init__(self, fire_config: FireConfiguration, agent_config: AgentConfiguration, reward_config: RewardConfiguration) -> None:
        """Initialize the agent with configuration parameters."""
        self.fire_config = fire_config
        self.agent_config = agent_config
        self.reward_config = reward_config

    def act(self, action_space) -> List[List[int]]:
        """
        Select actions based on the current state of the environment.

        Args:
            action_space: Available action space for the agent.
        
        Returns:
            List of actions for each agent in each parallel environment.
        """
        actions = []
        # Loop through each environment's state
        for state in action_space:
            env_actions = []
            for agent_index, agent_pos in enumerate(state['agents']):
                # Retrieve agent's allowable suppressant and equipment condition
                agent_suppressants = state['suppressants'][agent_index]
                agent_equipment = state['equipment'][agent_index]
                
                best_fire_idx = -1
                best_fire_intensity = -1

                # Explore fires within the agent's attack range
                for fire_idx, fire_intensity in enumerate(state['intensity'].flatten()):
                    fire_y, fire_x = divmod(fire_idx, state['fires'].shape[2])
                    
                    in_range = self.is_within_range(agent_pos, (fire_y, fire_x), self.agent_config.attack_range)
                    has_suppressants = agent_suppressants > 0
                    
                    if in_range and has_suppressants and fire_intensity > best_fire_intensity:
                        best_fire_idx = fire_idx
                        best_fire_intensity = fire_intensity

                # Decide action: Attack strongest fire in range or do nothing
                if best_fire_idx != -1:
                    attack_type = 0  # Use suppressant
                else:
                    best_fire_idx = 0  # Ensure valid index, even if no fire to target
                    attack_type = -1  # No operation

                env_actions.append([best_fire_idx, attack_type])

            actions.append(env_actions)

        return actions

    def observe(self, observation) -> None:
        """
        Observe the environment to update agent's internal state if necessary.

        Args:
            observation: Current observation from the environment.
        """
        # No persistent state kept by this simple agent, so nothing to do in observe
        pass

    def is_within_range(self, agent_pos, fire_pos, attack_range):
        """Determine if the fire is within the agent's attack range."""
        return (abs(agent_pos[0] - fire_pos[0]) <= attack_range and
                abs(agent_pos[1] - fire_pos[1]) <= attack_range)
