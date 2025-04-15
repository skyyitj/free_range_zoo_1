def single_agent_policy(
    # Agent's own state
    agent_pos: Tuple[float, float],
    agent_fire_reduction_power: float,
    agent_suppressant_num: float,

    # Other agents' states
    other_agents_pos: List[Tuple[float, float]],

    # Task information
    fire_pos: List[Tuple[float, float]],
    fire_levels: List[float],
    fire_intensities: List[float],
) -> int:
    """
    Determines the best action for an agent in the wildfire environment.
    """
    
    # Step 1: Calculate distance between agent and fire tasks
    def calculate_distance(agent_pos, fire_pos):
        return ((agent_pos[0] - fire_pos[0])**2 + (agent_pos[1] - fire_pos[1])**2) ** 0.5

    # Step 2: Create a list of fires with additional information like distances and suppression requirements
    fire_tasks = []
    for i, fire in enumerate(fire_pos):
        fire_tasks.append({
            "index": i,
            "position": fire,
            "level": fire_levels[i],
            "intensity": fire_intensities[i],
            "distance": calculate_distance(agent_pos, fire),
            "can_go_out_naturally": fire_levels[i] > 75  # Fires with levels > 75 will go out on their own
        })
    
    # Step 3: Sort fire tasks by a combination of urgency (fire level * intensity) and distance
    fire_tasks.sort(key=lambda x: (-x["level"] * x["intensity"], x["distance"]))

    # Step 4: Try to choose the most urgent fire to handle while respecting suppressant availability
    for fire_task in fire_tasks:
        # If fire is too high-level and will go out naturally, skip it
        if fire_task["can_go_out_naturally"]:
            continue
        
        # Check if agent has enough suppressant to manage the fire task
        if agent_suppressant_num >= fire_task["intensity"]:
            # Apply suppressant to fire
            return fire_task["index"]
    
    # Step 5: If no fires are chosen (e.g., due to suppressant limitations), return a neutral action
    return -1  # No action if no fire can be addressed