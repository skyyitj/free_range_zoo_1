def act(self, action_space):
    try:
        # Ensure maximum_index is valid in index_map
        maximum_index = self.get_action_index(action_space)
        if maximum_index not in self.index_map:
            raise ValueError(f"Maximum index {maximum_index} not found in index_map.")
        self.actions[batch, 0] = self.index_map[maximum_index]
    except Exception as e:
        # Log the error and handle it gracefully
        print(f"Error in act method: {e}")
        self.actions[batch, 0] = 0  # Default action, or some fallback action