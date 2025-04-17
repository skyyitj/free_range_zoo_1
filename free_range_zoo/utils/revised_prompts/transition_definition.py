class CapacityTransition(nn.Module):
    def __init__(self, agent_shape: Tuple, stochastic_switch: bool, tank_switch_probability: float,
                 possible_capacities: torch.Tensor, capacity_probabilities: torch.Tensor):
        super().__init__()

        self.register_buffer('stochastic_switch', torch.tensor(stochastic_switch, dtype=torch.bool))
        self.register_buffer('tank_switch_probability', torch.tensor(tank_switch_probability, dtype=torch.float32))
        self.register_buffer('possible_capacities', possible_capacities)
        self.register_buffer('capacity_probabilities', torch.cumsum(capacity_probabilities, dim=0))
        self.register_buffer('tank_switches', torch.zeros(agent_shape, dtype=torch.bool))

    def _reset_buffers(self) -> None:
        self.tank_switches.fill_(False)

    def forward(self, state: WildfireState, targets: torch.Tensor, randomness_source: torch.Tensor) -> WildfireState:
        self._reset_buffers()

        size_indices = torch.bucketize(randomness_source[0], self.capacity_probabilities)
        new_maximums = self.possible_capacities[size_indices]

        if self.stochastic_switch:
            self.tank_switches[targets] = randomness_source[1][targets] < self.tank_switch_probability
        else:
            self.tank_switches[targets] = True

        bonuses = state.suppressants - state.capacity

        state.capacity[self.tank_switches] = new_maximums[self.tank_switches]
        state.suppressants[self.tank_switches] = new_maximums[self.tank_switches]
        state.suppressants[self.tank_switches] += bonuses[self.tank_switches]

        return state

class EquipmentTransition(nn.Module):
    def __init__(self, equipment_states: torch.Tensor, stochastic_repair: bool, repair_probability: float,
                 stochastic_degrade: bool, degrade_probability: float, critical_error: bool, critical_error_probability: float):
        super().__init__()

        self.register_buffer('equipment_states', equipment_states)
        self.register_buffer('stochastic_repair', torch.tensor(stochastic_repair, dtype=torch.bool))
        self.register_buffer('repair_probability', torch.tensor(repair_probability, dtype=torch.float32))
        self.register_buffer('stochastic_degrade', torch.tensor(stochastic_degrade, dtype=torch.bool))
        self.register_buffer('degrade_probability', torch.tensor(degrade_probability, dtype=torch.float32))
        self.register_buffer('critical_error', torch.tensor(critical_error, dtype=torch.bool))
        self.register_buffer('critical_error_probability', torch.tensor(critical_error_probability, dtype=torch.float32))

    @torch.no_grad()
    def forward(self, state: WildfireState, randomness_source: torch.Tensor) -> WildfireState:
        pristine = state.equipment == (self.equipment_states.shape[0] - 1)
        damaged = state.equipment == 0
        intermediate = torch.logical_and(torch.logical_not(pristine), torch.logical_not(damaged))

        if self.stochastic_repair:
            repairs = torch.logical_and(damaged, randomness_source < self.repair_probability)
        else:
            repairs = damaged

        state.equipment[repairs] = self.equipment_states.shape[0] - 1

        if self.critical_error:
            criticals = torch.logical_and(pristine, randomness_source < self.critical_error_probability)
            state.equipment[criticals] = 0

        if self.stochastic_degrade:
            degrades = torch.logical_and(torch.logical_or(pristine, intermediate), randomness_source < self.degrade_probability)
        else:
            degrades = torch.logical_or(intermediate, pristine)

        if self.critical_error:
            degrades = torch.logical_and(degrades, torch.logical_not(criticals))

        state.equipment[degrades] -= 1

        return state
class FireDecreaseTransition(nn.Module):
    def __init__(self, fire_shape: Tuple, stochastic_decrease: bool, decrease_probability: float,
                 extra_power_decrease_bonus: float):
        super().__init__()

        self.register_buffer('stochastic_decrease', torch.tensor(stochastic_decrease, dtype=torch.bool))
        self.register_buffer('decrease_probability', torch.tensor(decrease_probability, dtype=torch.float32))
        self.register_buffer('extra_power_decrease_bonus', torch.tensor(extra_power_decrease_bonus, dtype=torch.float32))

        self.register_buffer('decrease_probabilities', torch.zeros(fire_shape, dtype=torch.float32))
    def _reset_buffers(self):
        """
        Reset the transition buffers
        """
        self.decrease_probabilities.fill_(0.0)
    def forward(self,
                state: WildfireState,
                attack_counts: torch.Tensor,
                randomness_source: torch.Tensor,
                return_put_out: bool = False) -> Union[WildfireState, Tuple[WildfireState, torch.Tensor]]:
        self._reset_buffers()
        required_suppressants = torch.where(state.fires >= 0, state.fires, torch.zeros_like(state.fires))
        attack_difference = required_suppressants - attack_counts

        lit_tiles = torch.logical_and(state.fires > 0, state.intensity > 0)
        suppressant_needs_met = torch.logical_and(attack_difference <= 0, lit_tiles)

        self.decrease_probabilities[:, :, :] = 0.0

        if self.stochastic_decrease:
            self.decrease_probabilities[suppressant_needs_met] = self.decrease_probability + \
                -1 * attack_difference[suppressant_needs_met] * self.extra_power_decrease_bonus
        else:
            self.decrease_probabilities[suppressant_needs_met] = 1.0

        self.decrease_probabilities = torch.clamp(self.decrease_probabilities, 0, 1)
        fire_decrease_mask = randomness_source < self.decrease_probabilities

        state.intensity[fire_decrease_mask] -= 1
        just_put_out = torch.logical_and(fire_decrease_mask, state.intensity <= 0)
        state.fires[just_put_out] *= -1
        state.fuel[just_put_out] -= 1

        if return_put_out:
            return state, just_put_out

        return state
 class FireIncreaseTransition(nn.Module):
    def __init__(self, fire_shape: Tuple, fire_states: int, stochastic_increase: bool, intensity_increase_probability: float,
                 stochastic_burnouts: bool, burnout_probability: float):
        super().__init__()

        self.register_buffer('almost_burnout_state', torch.tensor(fire_states - 2, dtype=torch.int32))
        self.register_buffer('burnout_state', torch.tensor(fire_states - 1, dtype=torch.int32))

        self.register_buffer('stochastic_increase', torch.tensor(stochastic_increase, dtype=torch.bool))
        self.register_buffer('intensity_increase_probability', torch.tensor(intensity_increase_probability, dtype=torch.float32))
        self.register_buffer('stochastic_burnouts', torch.tensor(stochastic_burnouts, dtype=torch.bool))
        self.register_buffer('burnout_probability', torch.tensor(burnout_probability, dtype=torch.float32))

        self.register_buffer('increase_probabilities', torch.zeros(fire_shape, dtype=torch.float32))

    def _reset_buffers(self):
        self.increase_probabilities.fill_(0.0)

    @torch.no_grad()
    def forward(self,
                state: WildfireState,
                attack_counts: torch.Tensor,
                randomness_source: torch.Tensor,
                return_burned_out: bool = False) -> Union[WildfireState, Tuple[WildfireState, torch.Tensor]]:
        self._reset_buffers()

        required_suppressants = torch.where(state.fires >= 0, state.fires, torch.zeros_like(state.fires))
        attack_difference = required_suppressants - attack_counts

        lit_tiles = torch.logical_and(state.fires > 0, state.intensity > 0)
        suppressant_needs_unmet = torch.logical_and(attack_difference > 0, lit_tiles)

        almost_burnouts = torch.logical_and(suppressant_needs_unmet, state.intensity == self.almost_burnout_state)
        increasing = torch.logical_and(suppressant_needs_unmet, ~almost_burnouts)

        if self.stochastic_increase:
            self.increase_probabilities[increasing] = self.intensity_increase_probability
        else:
            self.increase_probabilities[increasing] = 1.0

        if self.stochastic_burnouts:
            self.increase_probabilities[almost_burnouts] = self.burnout_probability
        else:
            self.increase_probabilities[almost_burnouts] = self.intensity_increase_probability

        self.increase_probabilities = torch.clamp(self.increase_probabilities, 0, 1)
        fire_increase_mask = randomness_source < self.increase_probabilities

        state.intensity[fire_increase_mask] += 1

        just_burned_out = torch.logical_and(fire_increase_mask, state.intensity >= self.burnout_state)

        state.fires[just_burned_out] *= -1
        state.fuel[just_burned_out] = 0

        if return_burned_out:
            return state, just_burned_out

        return state
class FireSpreadTransition(nn.Module):
    def __init__(self, fire_spread_weights: torch.Tensor, ignition_temperatures: torch.Tensor, use_fire_fuel: bool):
        super().__init__()

        self.register_buffer('ignition_temperatures', ignition_temperatures)
        self.register_buffer('use_fire_fuel', torch.tensor(use_fire_fuel, dtype=torch.bool))

        self.fire_spread_filter = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.fire_spread_filter.weight.data = fire_spread_weights

    @torch.no_grad()
    def forward(self, state: WildfireState, randomness_source: torch.Tensor) -> WildfireState:
        lit = torch.logical_and(state.fires > 0, state.intensity > 0)
        lit = lit.to(torch.float32).unsqueeze(1)
        fire_spread_probabilities = self.fire_spread_filter(lit).squeeze(1)

        unlit_tiles = torch.logical_and(state.fires < 0, state.intensity == 0)
        if self.use_fire_fuel:
            unlit_tiles = torch.logical_and(unlit_tiles, state.fuel > 0)

        fire_spread_probabilities[~unlit_tiles] = 0
        fire_spread_mask = randomness_source < fire_spread_probabilities

        state.fires[fire_spread_mask] *= -1
        state.intensity[fire_spread_mask] = self.ignition_temperatures.expand(state.fires.shape[0], -1, -1)[fire_spread_mask]

        return state
class SuppressantDecreaseTransition(nn.Module):
    def __init__(self, agent_shape: Tuple, stochastic_decrease: bool, decrease_probability: float):
        super().__init__()

        self.register_buffer('stochastic_decrease', torch.tensor(stochastic_decrease, dtype=torch.bool))
        self.register_buffer('decrease_probability', torch.tensor(decrease_probability, dtype=torch.float32))

        self.register_buffer('decrease_mask', torch.zeros(agent_shape, dtype=torch.bool))

    def _reset_buffers(self):
        self.decrease_mask.fill_(False)

    @torch.no_grad()
    def forward(self,
                state: WildfireState,
                used_suppressants: torch.Tensor,
                randomness_source: torch.Tensor,
                return_decreased: bool = False) -> WildfireState:

        self._reset_buffers()

        self.decrease_mask = used_suppressants
        if self.stochastic_decrease:
            self.decrease_mask = torch.logical_and(self.decrease_mask, randomness_source < self.decrease_probability)

        state.suppressants = torch.where(self.decrease_mask, state.suppressants - 1, state.suppressants)
        state.suppressants = torch.clamp(state.suppressants, min=0)

        if return_decreased:
            return state, self.decrease_mask

        return state
class SuppressantRefillTransition(nn.Module):
    def __init__(self, agent_shape: Tuple, stochastic_refill: bool, refill_probability: float, equipment_bonuses: torch.Tensor):
        super().__init__()

        self.register_buffer('stochastic_refill', torch.tensor(stochastic_refill, dtype=torch.bool))
        self.register_buffer('refill_probability', torch.tensor(refill_probability, dtype=torch.float32))
        self.register_buffer('equipment_bonuses', equipment_bonuses)

        self.register_buffer('increase_mask', torch.zeros(agent_shape, dtype=torch.bool))

    def _reset_buffers(self):
        """Reset the transition buffers."""
        self.increase_mask[:, :] = False

    @torch.no_grad()
    def forward(
        self,
        state: WildfireState,
        refilled_suppressants: torch.Tensor,
        randomness_source: torch.Tensor,
        return_increased: bool = False,
    ) -> WildfireState:
        self._reset_buffers()

        self.increase_mask[refilled_suppressants] = True
        if self.stochastic_refill:
            self.increase_mask = torch.logical_and(self.increase_mask, randomness_source < self.refill_probability)

        equipment_states = state.equipment.flatten().unsqueeze(1)
        equipment_bonuses = self.equipment_bonuses[equipment_states].reshape(self.increase_mask.shape[0], -1)

        state.suppressants[self.increase_mask] = state.capacity[self.increase_mask]
        state.suppressants[self.increase_mask] += equipment_bonuses[self.increase_mask]

        if return_increased:
            return state, self.increase_mask

        return state
