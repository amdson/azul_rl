# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 19:01:05 2022

@author: Ryan
"""
from Azul_Simulator import *
import numpy as np
rng = np.random.default_rng(6546547)

print("Welcome to the Azul Simulator!")
print()
state = init_board()
state = get_next_state(state, Action_Indices.NOOP.value, rng)
print(state)
disp_board(state)
print()
computer_turn = False
end = False

while not end:
    if computer_turn:
        print("Computer's turn...")
    else:
        print("Player's turn...")
    print("Enter a move: ")
    
    # Flip the state to put all player data in the current player section
    if not computer_turn:
        state = flip_board(state)
    
    # Get action from user and call simulator
    invalid_action_or_input = True
    while invalid_action_or_input:
        input_string = input()
        state, error = take_action_from_string(state, input_string, rng)

        # Validate input_string
        if error:
            print("Enter a different move: ")
            continue
        invalid_action_or_input = False
        
        # Handle game end condition
        reward, end = get_reward(state)
        if end and reward == -1:
            if computer_turn:
                print()
                print("Computer wins!!!")
            else:
                print()
                print("Player wins!!!")
        elif end and reward == 1:
            if computer_turn:
                print()
                print("Player wins!!!")
            else:
                print()
                print("Computer wins!!!")
        
        # Reset state for display purposes
        if not computer_turn:
                state = flip_board(state)
        disp_board(state)
        print()
        
        # Check if we need to do a noop
        if state[State_Indices.NOOP.value] == 1:
            state = get_next_state(state, Action_Indices.NOOP.value, rng)
        else:
            computer_turn = not computer_turn         
