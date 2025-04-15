def single_agent_policy(
       agent_pos: Tuple[float, float],
       agent_fire_reduction_power: float,
       agent_suppressant_num: float,
       other_agents_pos: List[Tuple[float, float]],
       fire_pos: List[Tuple[float, float]],
       fire_levels: List[float],
       fire_intensities: List[float]
   ) -> int:
       # Your policy logic