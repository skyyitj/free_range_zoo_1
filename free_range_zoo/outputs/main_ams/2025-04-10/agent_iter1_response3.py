def act(self, action_space):
    # Ensure we are selecting a valid action from the action space
    maximum_index = self.select_action(action_space)
    
    # Check if the maximum_index exists in index_map
    if maximum_index not in self.index_map:
        # Handle the case where the index is invalid
        print(f"Error: index {maximum_index} not found in index_map.")
        # Fallback: Choose the first valid index or a default action
        maximum_index = 0
    
    # Retrieve the action from the index map
    self.actions[batch, 0] = self.index_map[maximum_index]
    return self.actions[batch, 0]