def single_agent_policy(
    agent_pos: Tuple[float, float],
    agent_fire_reduction_power: float,
    agent_suppressant_num: float,
    other_agents_pos: List[Tuple[float, float]],
    fire_pos: List[Tuple[float, float]],
    fire_levels: List[float],
    fire_intensities: List[float],
) -> int:
    if agent_suppressant_num <= 0 or not fire_pos:
        return -1  # No suppressant or no tasks available

    max_auto_extinguish_level = 10.0  # If fire gets here, agent gets penalized

    # Prioritize: fires that can be helped (not too large), close, and high intensity
    best_score = float("-inf")
    best_index = -1

    for i, (f_pos, f_level, f_intensity) in enumerate(zip(fire_pos, fire_levels, fire_intensities)):
        if f_level >= max_auto_extinguish_level or f_level <= 0:
            continue  # Skip fires that are going to extinguish naturally or are already gone

        # Heuristic: Prefer closer fires with higher level and intensity
        distance = ((agent_pos[0] - f_pos[0]) ** 2 + (agent_pos[1] - f_pos[1]) ** 2) ** 0.5 + 1e-5
        score = (f_intensity + f_level) / distance

        if score > best_score:
            best_score = score
            best_index = i

    return best_index if best_index != -1 else 0  # fallback to avoid KeyError