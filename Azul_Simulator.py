# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 14:23:02 2022

@author: Ryan Huntley
"""

from enum import Enum
from colorama import Fore, Back, init
import numpy as np

five_by_five_offset = (4 * 5) + 1 # Used for constructing ranges

Board = np.array([[0, 1, 2, 3, 4],
                  [4, 0, 1, 2, 3],
                  [3, 4, 0, 1, 2],
                  [2, 3, 4, 0, 1],
                  [1, 2, 3, 4, 0]])

class Tile_Offsets(Enum):   
    B = 0 # Blue
    Y = 1 # Yellow flower
    R = 2 # Red
    D = 3 # Black flower (the D stands for "dark")
    W = 4 # White flower
    S = 5 # Starting tile
    
class State_Indices(Enum):
    NOOP = 0                                            # 1
    SCORE = 1                                           # 1
    SCORE_OPP = 2                                       # 1
    SCORE_DIFF = 3                                      # 1
    BAG = 4                                             # 5
    DISCARD = 9                                         # 5
    FACTORIES = range(14, 14 + five_by_five_offset, 5)  # 5 x 5
    CENTER = 39                                         # 6
    BOARD = 45                                          # 25
    ROWS = range(70, 70 + five_by_five_offset, 5)       # 5 x 5
    FLOOR = 95                                          # 6
    BOARD_OPP = 101                                     # 25
    ROWS_OPP = range(126, 126 + five_by_five_offset, 5) # 5 x 5
    FLOOR_OPP = 151                                     # 6

class Action_Indices(Enum):
    NOOP = 0       # 1
    CENTER = 1     # 5 x 6
    FACTORIES = 31 # 5 x 5 x 6

def get_expanded_bag(bag):
    """
    Inverse operation of "get_expanded_bag".
    
    Parameters
    ----------
    bag : np.array[int]
        Compact representation of a bag (used in state space).

    Returns
    -------
    expanded_bag : np.array[int]
        Expanded representation of a bag (used for sampling).
    """
    num_bag = np.sum(bag)
    expanded_bag = np.zeros(num_bag, dtype=np.int8)
    counter = 0
    for tile in range(5):
        for i in range(bag[tile]):
            expanded_bag[counter] = tile
            counter += 1
    return expanded_bag

def get_compact_bag(expanded_bag):
    """
    Inverse operation of "get_expanded_bag".
    
    Parameters
    ----------
    expanded_bag : np.array[int]
        Expanded representation of a bag (used for sampling).

    Returns
    -------
    bag : np.array[int]
        Compact representation of a bag (used in state space).
    """
    bag = np.zeros(5, dtype=np.int8)
    for tile in expanded_bag:
        bag[tile] = bag[tile] + 1
    return bag

def shuffle_tiles(tiles, rng):
    """
    Parameters
    ----------
    tiles : np.array[int]
        Indices [4, 14) of a state vector, representing the number of each tile
        type that can be found in the bag and in the discard.
    rng : np.random.Generator
        Global seeded generator object.

    Returns
    -------
    selected : np.array[int]
        List (in order) of all tiles selected.
    new_tiles : np.array[int]
        New tile vector representing updated state of the bag and the discard.
    """
    bag = tiles[:5]
    discard = tiles[5:]
    num_bag = np.sum(bag)
    num_discard = np.sum(discard)
    
    # We ran out of tiles (this should never happen)
    if num_bag == 0 and num_discard == 0:
        return None, None
    
    # Bag is empty, put all tiles from the discard back in the bag
    if num_bag == 0:
        discard = np.copy(bag)
        bag = np.copy(discard)
        num_bag = np.sum(bag)
        num_discard = np.sum(discard)
    
    # Transform compact representation of bag into extended representation for sampling
    expanded_bag = get_expanded_bag(bag)
            
    # Reorder bag_extended and sample by taking from the front
    rng.shuffle(expanded_bag)
    num_to_select = min(num_bag, 20)
    selected = expanded_bag[:num_to_select]
    expanded_bag = expanded_bag[num_to_select:]
    
    # Check if we need to shuffle in the discarded tiles
    if num_to_select < 20 and num_discard > 0:
        bag = np.copy(discard)
        discard = np.zeros(5, dtype=0)
        
        # Repeat earlier process for the remaining tiles
        expanded_bag = get_expanded_bag(bag)
        num_to_select = 20 - num_to_select
        selected = np.concatenate(selected, expanded_bag[:num_to_select])
        expanded_bag = expanded_bag[num_to_select:]
    
    bag = get_compact_bag(expanded_bag)
        
    # Update current tile representation and return
    new_tiles = np.concatenate((bag, discard))
    return selected, new_tiles   

def add_tiles_to_factories(state, expanded_bag):
    """
    Parameters
    ----------
    state : np.array[int]
        Representation (len 157) of the current state with empty factories.
    expanded_bag : np.array[int]
        List of tiles to add (in order) to the factories.

    Returns
    -------
    state : np.array[int]
        Representation (len 157) of the state with the tiles in 
        expanded_bag added.
    """
    counter = 0
    new_state = np.copy(state)
    
    # Iterate over factories
    for factory in (State_Indices.FACTORIES.value):
        # Iterate over tile_types
        for _ in range(4):
            index = factory + expanded_bag[counter]
            new_state[index] = new_state[index] + 1
            counter += 1
            
            # Check if we aren't fully filling up the factories
            if counter > len(expanded_bag):
                break
    
    # Add 1st player tile to the center and return
    new_state[State_Indices.CENTER.value + Tile_Offsets.S.value] = 1
    return new_state
    
def init_board(rng):
    """
    Parameters
    ----------
    rng : np.random.Generator
        Global seeded generator object.
    
    Returns
    -------
    state : np.array[int]
        Representation (len 157) of the original state.
    """
    tiles = np.array([20, 20, 20, 20, 20, 0, 0, 0, 0, 0])
    selected, new_tiles = shuffle_tiles(tiles, rng)
    state = np.zeros(157, dtype=int)
    state[State_Indices.BAG.value:State_Indices.FACTORIES.value[0]] = new_tiles
    state = add_tiles_to_factories(state, selected)    
    return state
     
def flip_board(state):
    """
    Parameters
    ----------
    state : vector
        Representation (len 157) of the current state.

    Returns
    -------
    state : vector
        Representation (len 157) of the state with all opponent information
        in the corresponding position for the player and vice versa.
    """
    board = State_Indices.BOARD.value
    board_opp = State_Indices.BOARD_OPP.value
    new_state = np.copy(state)
    new_state[board:board_opp] = np.copy(state[board_opp:])
    new_state[board_opp:] = np.copy(state[board:board_opp])
    new_state[State_Indices.SCORE.value] = state[State_Indices.SCORE_OPP.value]
    new_state[State_Indices.SCORE_OPP.value] = state[State_Indices.SCORE.value]
    new_state[State_Indices.SCORE_DIFF.value] *= -1
    return new_state

def disp_board(state):
    """
    Assumes that the "current player" is always the computer. You must call
    flip_state beforehand if this is not true.
    
    Parameters
    ----------
    state : np.array[int]
        Representation (len 157) of the original state.

    Returns
    -------
    None.
    """
    # Initialize colorama module
    init(strip=False)
    
    # Construct a dictionary mapping tile_offsets to their display letter and highlighted variant
    get_letter = {}
    get_highlight = {}
    colors = [Fore.BLUE, Fore.YELLOW, Fore.RED, Fore.BLACK, Fore.WHITE, Fore.BLACK]
    colors_h = [Back.BLUE, Back.YELLOW, Back.RED, Back.BLACK, Back.WHITE, Back.BLACK]
    counter = 0
    for data in Tile_Offsets:
        get_letter[data.value] = colors[counter] + data.name + Fore.RESET
        get_highlight[data.value] = colors_h[counter] + data.name + Back.RESET
        counter += 1
    
    # Utils
    big_line = "#" * 46
    line = "-" * 46
    print(big_line)
    
    # TABLE
    print("TABLE:")
    print(line)
    
    # Center
    center_string = "Center:     "
    not_present = "(no "
    extra_space = " " * 24
    
    if state[State_Indices.CENTER.value + Tile_Offsets.S.value] == 1:
        center_string += "*1*"
        extra_space = extra_space[:-3]
        
    for i in range(5):
        n = state[State_Indices.CENTER.value + i]
        if n == 0:
            not_present += get_letter[i]
        else:
            center_string += (get_letter[i] * n)
            extra_space = extra_space[:-1 * n]
            
    center_string += extra_space + not_present + ")"
    print(center_string)
    
    # Factories
    for i in range(5):
        factory_string = "Factory #" + str(i + 1) + ": "
        not_present = "(no "
        extra_space = " " * 24
        
        for j in range(5):
            n = state[State_Indices.FACTORIES.value[i] + j]
            if n == 0:
                not_present += get_letter[j]
            else:
                factory_string += (get_letter[j] * n)
                extra_space = extra_space[:-1 * n]
        
        factory_string += extra_space + not_present + ")"
        print(factory_string)
        
    # COMPUTER
    print(line)
    print()
    print("COMPUTER: (score - " + str(state[State_Indices.SCORE.value]) + ")")
    print(line)
    
    # Rows
    for i in range(5):
        spaces = "  " * (4 - i)
        underscores = "_ " * (i + 1)
        
        for j in range(5):
            n = state[State_Indices.ROWS.value[i] + j]
            if n != 0:
                underscores = underscores[:-2 * n] + (n * (get_letter[j] + " "))
        
        board_row = ""
        for j in range(5):
            if state[State_Indices.BOARD.value + (5 * i) + j] == 1:
                board_row += get_highlight[Board[i][j]] + " "
            else:
                board_row += get_letter[Board[i][j]] + " "
        
        print(spaces + underscores + " | " + board_row)
        
    # Floor
    floor_string = "Floor: "
    if state[State_Indices.FLOOR.value + Tile_Offsets.S.value] == 1:
        floor_string += "*1*"
        
    for i in range(5):
        n = state[State_Indices.FLOOR.value + i]
        if n != 0:
            floor_string += (get_letter[i] * n)
    
    print()
    print(floor_string)
        
    # PLAYER
    print(line)
    print()
    print("PLAYER: (score - " + str(state[State_Indices.SCORE_OPP.value]) + ")")
    print(line)
    
    # Rows
    for i in range(5):
        spaces = "  " * (4 - i)
        underscores = "_ " * (i + 1)
        
        for j in range(5):
            n = state[State_Indices.ROWS_OPP.value[i] + j]
            if n != 0:
                underscores = underscores[:-2 * n] + (n * (get_letter[j] + " "))
        
        board_row = ""
        for j in range(5):
            if state[State_Indices.BOARD_OPP.value + (5 * i) + j] == 1:
                board_row += get_highlight[Board[i][j]] + " "
            else:
                board_row += get_letter[Board[i][j]] + " "
        
        print(spaces + underscores + " | " + board_row)
    
    # Floor
    floor_string = "Floor: "
    if state[State_Indices.FLOOR_OPP.value + Tile_Offsets.S.value] == 1:
        floor_string += "*1*"
        
    for i in range(5):
        n = state[State_Indices.FLOOR_OPP.value + i]
        if n != 0:
            floor_string += (get_letter[i] * n)
    
    print()
    print(floor_string)
    
    # Final lines
    print(line)
    print(big_line)

def score_line(line, pos, horizontal):
    """
    Scores a line of five tiles (could be a row or a column) given a tile that
    was recently added at the given position.

    Parameters
    ----------
    line : np.array
        Boolean array (len 5) representing up to five tiles. 
    pos : int
        The most recently added tile to that array.
    horizontal : bool
        Whether or not the given line is a horizontal line (row)

    Returns
    -------
    score : int
        The score of the line.
    """
    assert line[pos] == 1, "'pos' does not correspond to a tile in 'line'"
    score = 1
    
    # Check tiles before pos
    for i in reversed(range(0, pos)):
        if line[i] == 1:
            score += 1
        else:
            break
        
    # Check tiles after pos
    for i in range(pos + 1, 5):
        if line[i] == 1:
            score += 1
        else:
            break
    
    # If just one tile, revert score back to 0
    if score == 1:
        score = 0
    
    # Check for completed row/column
    if np.sum(line) == 5:
        if horizontal:
            score += 2
        else:
            score += 7
            
    return score

def check_fifth_tile(tile, board):
    """
    Parameters
    ----------
    tile : int
        Number corresponding to a given tile.
    board : np.array
        State of the board (len 25).

    Returns
    -------
    fifth_tile : bool
        Whether or not the fifth tile of a type was just placed.
    """
    assert len(board) == 25, "'board' should be an array of length 25"
    
    flattened_board = Board.flatten()
    mask = np.where(flattened_board == tile, board, 0)
    if np.sum(mask) == 5:
        return True
    else:
        return False
    
def get_floor_penalty(tiles):
    """
    Parameters
    ----------
    tiles : np.array
        State of the tiles on the floor (len 6).

    Returns
    -------
    penalty : int
        Total number of points lost for all tiles on the floor.
    """
    assert len(tiles) == 6, "'tiles' should be an array of length 6"
    
    first_five = np.array([1, 1, 2, 2, 2])
    num_on_floor = np.sum(tiles)
    if num_on_floor <= 5:
        return np.sum(first_five[:num_on_floor])
    else:
        return np.sum(first_five) + ((num_on_floor - 5) * 3)
    
def shift_and_score(state):
    """
    Moves all tiles from the rows to their corresponding places on the board
    for both players. Then computes the change in score for both players and
    updates the state.
    
    Parameters
    ----------
    state : np.array[int]
        Representation (len 157) of the current state.

    Returns
    -------
    new_state : np.array[int]
        Representation (len 157) of the state after scoring.
    end : bool
        Whether or not a game ending condition has been met
    """
    score = 0
    score_opp = 0
    new_state = np.copy(state)
    
    # Current Player - iterate over rows
    for i in range(5):
        row_start = State_Indices.ROWS.value[i]
        
        # Iterate over possible tiles in each row
        for j in range(row_start, row_start + 5):
            # We found a tile (corresponding to offset j) within row i
            if new_state[j] == i + 1:
                board_start = State_Indices.BOARD.value
                board_row_start = board_start + (i * 5)
                
                # Find the same tile in the corresponding row of the board
                for k in range(board_row_start, board_row_start + 5):
                    if Board[i][k - board_row_start] == j - row_start:
                        # Check to make sure there isn't already a tile there
                        error_message = "Tile #" + str(j - row_start) + " should not have been in row " + str(i)
                        assert new_state[k] == 0, error_message
                        
                        # Update board
                        new_state[k] = 1
                        
                        # Update score based on rows and columns
                        current_board = np.reshape(new_state[board_start:board_start + 25], (5, 5))
                        add_to_score1 = score_line(current_board[i], k - board_row_start, True)
                        score += add_to_score1
                        add_to_score2 = score_line(current_board[:, k - board_row_start], i, False)
                        score += add_to_score2
                        
                        # Lone tile gets 1 point
                        if add_to_score1 == 0 and add_to_score2 == 0:
                            score += 1
                        
                        # Check if the fifth tile of a color was just placed
                        if check_fifth_tile(j - row_start, new_state[board_start:board_start + 25]):
                            score += 10
                        
                # Clear the current row and move on to the next one
                new_state[j] = 0
                break
            
    # Subtract the floor penalty for the current player
    floor_start = State_Indices.FLOOR.value
    score -= get_floor_penalty(new_state[floor_start:floor_start + 6])
    new_state[floor_start:floor_start + 6] = np.zeros(6, dtype=int)
    new_state[State_Indices.SCORE.value] += score
    
    # Opponent - iterate over rows
    for i in range(5):
        row_start = State_Indices.ROWS_OPP.value[i]
        
        # Iterate over possible tiles in each row
        for j in range(row_start, row_start + 5):
            # We found a tile (corresponding to offset j) within row i
            if new_state[j] == i + 1:
                board_start = State_Indices.BOARD_OPP.value
                board_row_start = board_start + (i * 5)
                
                # Find the same tile in the corresponding row of the board
                for k in range(board_row_start, board_row_start + 5):
                    if Board[i][k - board_row_start] == j - row_start:
                        # Check to make sure there isn't already a tile there
                        error_message = "Tile #" + str(j - row_start) + " should not have been in row " + str(i)
                        assert new_state[k] == 0, error_message
                        
                        # Update board
                        new_state[k] = 1
                        
                        # Update score based on rows and columns
                        current_board = np.reshape(np.copy(new_state[board_start:board_start + 25]), (5, 5))
                        add_to_score1 = score_line(current_board[i], k - board_row_start, True)
                        score_opp += add_to_score1
                        add_to_score2 = score_line(current_board[:, k - board_row_start], i, False)
                        score_opp += add_to_score2
                        
                        # Lone tile gets 1 point
                        if add_to_score1 == 0 and add_to_score2 == 0:
                            score_opp += 1
                        
                        # Check if the fifth tile of a color was just placed
                        if check_fifth_tile(j - row_start, new_state[board_start:board_start + 25]):
                            score_opp += 10
                        
                # Clear the current row and move on to the next one
                new_state[j] = 0
                break
            
    # Subtract the floor penalty for the current player
    floor_start = State_Indices.FLOOR_OPP.value
    score_opp -= get_floor_penalty(new_state[floor_start:floor_start + 6])
    new_state[floor_start:floor_start + 6] = np.zeros(6, dtype=int)
    new_state[State_Indices.SCORE_OPP.value] += score_opp

    # Compute difference in scores and return
    new_state[State_Indices.SCORE_DIFF.value] = new_state[State_Indices.SCORE.value] - new_state[State_Indices.SCORE_OPP.value]
    return new_state        
    
def get_next_state(state, action, rng):
    """
    Parameters
    ----------
    state : np.array[int]
        Representation (len 157) of the original state.
    action : int
        Index corresponding to the desired action.
    rng : np.random.Generator
        Global seeded generator object.

    Returns
    -------
    new_state : np.array[int]
        Representation (len 157) of the subsequent state (after taking action).
    """
    new_state = np.copy(state)
    
    # Check for noop action
    noop = State_Indices.NOOP.value
    if action == Action_Indices.NOOP.value:
        if state[noop] == 1:
            new_state[noop] = 0
            return new_state
        # Noop is not a valid action
        else:
            raise RuntimeError("An illegal action was attempted")
    # Must take a noop action
    elif state[noop] == 1:
        raise RuntimeError("An illegal action was attempted")
        
    # Deal with center action
    elif action < Action_Indices.FACTORIES.value:
        offset = action - Action_Indices.CENTER.value
        tile = offset // 6
        row_num = offset % 6
        
        # Check that the center does contain one or more corresponding tiles
        center_index = State_Indices.CENTER.value + tile
        if state[center_index] == 0:
            raise RuntimeError("An illegal action was attempted")
        
        # Move tiles directly to the floor
        if row_num == 5:
            new_state[center_index] = 0
            new_state[State_Indices.FLOOR.value + tile] += state[center_index]
        else:
            # Check to make sure the tiles can be placed in the row
            row = State_Indices.ROWS.value[row_num]
            for i in range(5):
                # Another type of tile is in this row
                if state[row + i] > 0 and i != tile:
                    raise RuntimeError("An illegal action was attempted")
                # This row is already full
                elif state[row + i] == row_num + 1:
                    raise RuntimeError("An illegal action was attempted")
                
            # Check to make sure the corresponding spot on the board isn't filled
            board_start = State_Indices.BOARD.value
            current_board = np.reshape(np.copy(new_state[board_start:board_start + 25]), (5, 5))
            for i in range(5):
                if Board[row_num][i] == tile and current_board[row_num][i] == 1:
                    raise RuntimeError("An illegal action was attempted")
                
            # Move tiles
            new_state[center_index] = 0
            num_in_row = state[row + tile]
            remaining_spaces = row_num + 1 - num_in_row
            add_to_row = min(state[center_index], remaining_spaces)
            add_to_floor = state[center_index] - add_to_row
            new_state[row + tile] += add_to_row
            new_state[State_Indices.FLOOR.value + tile] += add_to_floor
        
        # Move 1st player tile if necessary
        if state[State_Indices.CENTER.value + 5] == 1:
            new_state[State_Indices.CENTER.value + 5] = 0
            new_state[State_Indices.FLOOR.value + 5] = 1
                     
    # Deal with factory action
    else:
        offset = (action - Action_Indices.FACTORIES.value) % 30
        factory_num = (action - Action_Indices.FACTORIES.value) // 30
        tile = offset // 6
        row_num = offset % 6
        
        # Check that the factory does contain one or more corresponding tiles
        factory_index = State_Indices.FACTORIES.value[factory_num] + tile
        if state[factory_index] == 0:
            raise RuntimeError("An illegal action was attempted")
        
        # Move tiles directly to the floor
        if row_num == 5:
            new_state[factory_index] = 0
            new_state[State_Indices.FLOOR.value + tile] += state[factory_index]
        else:
            # Check to make sure the tiles can be placed in the row
            row = State_Indices.ROWS.value[row_num]
            for i in range(5):
                # Another type of tile is in this row
                if state[row + i] > 0 and i != tile:
                    raise RuntimeError("An illegal action was attempted")
                # This row is already full
                elif state[row + i] == row_num + 1:
                    raise RuntimeError("An illegal action was attempted")
                
            # Check to make sure the corresponding spot on the board isn't filled
            board_start = State_Indices.BOARD.value
            current_board = np.reshape(np.copy(new_state[board_start:board_start + 25]), (5, 5))
            for i in range(5):
                if Board[row_num][i] == tile and current_board[row_num][i] == 1:
                    raise RuntimeError("An illegal action was attempted")
                
            # Move tiles
            new_state[factory_index] = 0
            num_in_row = state[row + tile]
            remaining_spaces = row_num + 1 - num_in_row
            add_to_row = min(state[factory_index], remaining_spaces)
            add_to_floor = state[factory_index] - add_to_row
            new_state[row + tile] += add_to_row
            new_state[State_Indices.FLOOR.value + tile] += add_to_floor
        
        # Move excess tiles to center
        factory_start = factory_index - tile
        center_start = State_Indices.CENTER.value
        for i in range(5):
            new_state[center_start + i] += new_state[factory_start + i]
            new_state[factory_start + i] = 0
            
    # If necessary, score boards and prepare for the following round
    table_start = State_Indices.FACTORIES.value[0]
    table_end = State_Indices.BOARD.value
    if not np.any(new_state[table_start:table_end]):
        # Set noop flag if current player has the 1st player token
        if new_state[State_Indices.FLOOR.value + 6] == 1:
            new_state[noop] = 1
        
        # Score both boards
        new_state = shift_and_score(new_state)
        
        # Prepare for the next round
        selected, new_tiles = shuffle_tiles(np.copy(state[State_Indices.BAG.value:State_Indices.FACTORIES.value[0]]), rng)
        new_state[State_Indices.BAG.value:State_Indices.FACTORIES.value[0]] = new_tiles
        new_state = add_tiles_to_factories(new_state, selected)
        
    return new_state

def get_valid_mask(state):
    """
    Parameters
    ----------
    state : np.array[int]
        Representation (len 157) of the current state.

    Returns
    -------
    mask : np.array[int]
        Representation (len 181) of the action space where indices corresponding
        to legal actions are set to 1, and all others are set to 0
    """
    # Check if a noop is the only valid action
    if state[State_Indices.NOOP.value] == 1:
        mask = np.zeros(181)
        mask[0] = 1
        return mask
    
    # Construct 3d representation of action space with all actions initially legal
    mask = np.ones((6, 5, 6))
    board_start = State_Indices.BOARD.value
    current_board = np.reshape(np.copy(state[board_start:board_start + 25]), (5, 5))
    
    # Check which tiles can be taken from the center
    for tile in range(5):
        if state[State_Indices.CENTER.value + tile] == 0:
            mask[0, tile, :] = 0
            
    # Check which tiles can be taken from the factories
    for factory in range(5):
        for tile in range(5):
            if state[State_Indices.FACTORIES.value[factory] + tile] == 0: 
                mask[factory + 1, tile, :] = 0
                
    # Check each row individually
    for row in range(5):
        row_start = State_Indices.ROWS.value[row]
        total = np.sum(state[row_start:row_start + 5])
        
        # Row is empty, we only need to check the board
        if total == 0:
            for i in range(5):
                if current_board[row][i] == 1:
                    mask[:, Board[row][i], row] = 0

        # Row is full, we can zero out everything
        elif total == row + 1:
            mask[:, :, row] = 0
            
        # Row is partially full, we need to allow for more of the already
        # present tile to be added
        else:
            for tile in range(5):
                if state[row_start + tile] == 0:
                    mask[:, tile, row] = 0
                    
    # Reshape and add in another index at the front set to 0
    temp = np.reshape(mask, 180)
    mask = np.zeros(181)
    mask[1:] = temp
    return mask   

def take_action_from_string(state, input_string, rng):
    """
    Converts the 3-letter input string into an index corresponding to an action.
    The letters of the input string corresponding to the following...
        index 0 - (C, 1, 2, 3, 4, 5) where to take tiles from
        index 1 - (B, Y, R, D, W) which color tile to take
        index 2 - (1, 2, 3, 4, 5, F) which row to place the tiles in    
    Ex: 5B2 corresponds to taking all blue (B) tiles from factory #5 and placing
        them in the second row
    Ex: CWF corresponds to taking all white (W) tiles from the center and placing
        them on the floor

    Parameters
    ----------
    state : np.array[int]
        Representation (len 157) of the original state.
    input_string : string
        Input string corresponding to an action.
    rng : np.random.Generator
        Global seeded generator object.

    Returns
    -------
    new_state : np.array[int]
        Representation (len 157) of the subsequent state (after taking action)
    error : bool
        True if an action was not successfully executed 
    """
    index0_list = np.array(['C', '1', '2', '3', '4', '5'])
    index1_list = np.array(['B', 'Y', 'R', 'D', 'W'])
    index2_list = np.array(['1', '2', '3', '4', '5', 'F'])
    index = 1 # The 0th action index corresponds to a noop
    
    # Check for valid input_string
    try:
        assert len(input_string) == 3
        assert input_string[0] in index0_list
        assert input_string[1] in index1_list
        assert input_string[2] in index2_list
    except AssertionError:
        print("Invalid input string, pease try again.")
        return state, True
    
    # Index 0
    for i in range(6):
        if index0_list[i] == input_string[0]:
            index += 30 * i
            
    # Index 1
    for i in range(5):
        if index1_list[i] == input_string[1]:
            index += 6 * i
            
    # Index 2
    for i in range(6):
        if index2_list[i] == input_string[2]:
            index += i
        
    # Call 'take_action' and return
    if get_valid_mask(state)[index] == 0:
        print("That action is illegal, please select another action.")
        return state, True
    new_state = get_next_state(state, index, rng)
    return new_state, False
    
def get_reward(state):
    """
    Parameters
    ----------
    state : np.array[int]
        Representation (len 157) of the current state.

    Returns
    -------
    reward : int
        -1 on a win (to account for the board immediately being flipped) 1 on
        a loss, 0 otherwise
    end: bool
        Whether or not the game is over
    """
    reward = 0
    end = False

    # Check for end condition in current player's board
    board = state[State_Indices.BOARD.value:State_Indices.ROWS.value[0]]
    for row in board.reshape((5, 5)):
        if np.sum(row) == 5:
            end = True
            break
    
    # Check for end condition in opponent's board
    board = state[State_Indices.BOARD_OPP.value:State_Indices.ROWS_OPP.value[0]]
    for row in board.reshape((5, 5)):
        if np.sum(row) == 5:
            end = True
            break
    
    # Calculate reward if game will end
    if end:
        if state[State_Indices.SCORE_DIFF.value] > 0:
            reward = -1
        elif state[State_Indices.SCORE_DIFF.value] < 0:
            reward = 1
            
    return reward, end
