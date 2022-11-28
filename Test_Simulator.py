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
state = init_state(rng)
print_state(state)
print()
computer_turn = True
end = False

while not end:
    if computer_turn:
        print("Computer's turn...")
    else:
        print("Player's turn...")
    print("Enter a move: ")
    
    # Flip the state to put all player data in the current player section
    if not computer_turn:
        state = flip_state(state)
    
    # Get action from user and call simulator
    invalid_action_or_input = True
    while invalid_action_or_input:
        input_string = input()
        state, reward = take_action_from_string(state, input_string, rng)
        
        # Validate input_string
        if reward is None:
            print("Enter a different move: ")
            continue
        invalid_action_or_input = False
        
        # Handle game end condition
        if reward == 100000:
            if computer_turn:
                print()
                print("Computer wins!!!")
            else:
                print()
                print("Player wins!!!")
            end = True
        elif reward == -100000:
            if computer_turn:
                print()
                print("Player wins!!!")
            else:
                print()
                print("Computer wins!!!")
            end = True
        
        # Reset state for display purposes
        if not computer_turn:
                state = flip_state(state)
        print_state(state)
        print()
        
        # Check if we need to do a noop
        if state[State_Indices.NOOP.value] == 1:
            state = take_action(state, Action_Indices.NOOP.value, rng)
        else:
            computer_turn = not computer_turn         
