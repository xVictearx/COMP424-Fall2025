# Student agent: VICTOR XIE, QUEENIE CHEN, RICHARD WANG
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, execute_move, check_endgame, get_valid_moves


EVAL_WEIGHTS = {
    # board[1,1] == 3
    "big_x": {                ###MULTIPLIER VALUES TO BE TUNED###
        "score_diff":    1,   
        "gap_offender": 70,   # encourages attacking the opponent gaps
        "gap_defender": 60,   # encourages defending our own gaps
        "move_bonus":   40,   # jumps slightly punished (your move_bonus is 0 or -5)
        "mobility":      5,   # open so mobility important
        "opp_hole_bonus":5,
        "obstacle": []
    },

    # board[2,3] == 3 and board[1,3] != 3
    "plus1": {
        "score_diff":    1,
        "gap_offender": 80,   
        "gap_defender": 50,  
        "move_bonus":   40,  
        "mobility":      5,  
        "opp_hole_bonus":5,
        "obstacle": []
    },

    # board[1,3] == 3
    "plus2": {
        "score_diff":    1,
        "gap_offender": 60,   
        "gap_defender": 70,
        "move_bonus":   40,
        "mobility":      5,  
        "opp_hole_bonus":1,
        "obstacle": []
    },

    # board[2,2] == 3
    "point4": {
        "score_diff":     1,
        "gap_offender":  80,   
        "gap_defender":  75,   
        "move_bonus":    60,
        "mobility":       5,   
        "opp_hole_bonus":10,
        "obstacle": []
    },

    # board[1,2] == 3
    "circle": {
        "score_diff":    1,
        "gap_offender": 90, 
        "gap_defender": 30,
        "move_bonus":   60,
        "mobility":     50,  
        "opp_hole_bonus":2,
        "obstacle": []
        
    },

    # board[0,3] == 3
    "wall": {
        "score_diff":    1,
        "gap_offender": 90,   
        "gap_defender": 30,  
        "move_bonus":   50,  
        "mobility":    100,   
        "opp_hole_bonus":1,
        "obstacle": [
                        (0,2),      (0,4),
                        (1,2),      (1,4),
          (2,0), (2,1), (2,2),      (2,4), (2,5), (2,6),
                              
          (3,0), (3,1), (3,2),      (3,4), (3,5), (3,6),
                        (4,2),      (4,4),
                        (5,2),      (5,4),
                        (6,2),      (6,4)
                    ]
    },

    # board[0,2] == 3
    "watch_sides": {
        "score_diff":     1,
        "gap_offender":  80,   
        "gap_defender":  80,
        "move_bonus":    60,
        "mobility":       5,   
        "opp_hole_bonus":20,
        "obstacle": []
    },

    # else: empty / no obstacles
    "empty": {
        "score_diff":     1,
        "gap_offender": 100,  
        "gap_defender":  40,   
        "move_bonus":    30,   
        "mobility":       5,   
        "opp_hole_bonus":20,
        "obstacle": []
    },
}

@register_agent("student_agent")
class StudentAgent(Agent):
  """
  A class for your implementation. Feel free to use this class to
  add any helper functionalities needed for your agent.
  """
 
  def __init__(self):
    super(StudentAgent, self).__init__()
    self.name = "StudentAgent"
    self.turns = 0
    self.is_difficult = False
    self.dir_coords = [#this is a list of all 8 directions for gap_points()
        (-1,-1), (-1, 0), (-1, 1),
        ( 0,-1),          ( 0, 1),
        ( 1,-1), ( 1, 0), ( 1, 1)
    ]

  def step(self, chess_board, you, opp):
    
    start_time = time.time()
    
    legal_moves = get_valid_moves(chess_board, you)
    if not legal_moves:
      return None  # No valid moves available, pass turn
    
    
    a = float('-inf')
    b =  float('inf')
    best_score = float('-inf')
    best_move = None
    i = 2
   
    board_type = self.get_board_type(chess_board)
    move_set = []
    
    # Early game: focus on one type of move to make sure we have a good amount of pieces on board for some board types
    if self.turns < 3 and (board_type in ["wall", "watch_sides", "circle"]):
      self.is_difficult = True #bool for boards we deemed difficult, will look at abs values of jumps only (from testing)
      move_set = [self.split_moves(legal_moves)[0]] #only consider duplicate moves

    else:
      move_set = self.split_moves(legal_moves) #consider both types of moves, duplicate moves first

    ### Minimax with alpha-beta pruning ###
    for moves in move_set:
      for move in moves:
          
        sim_board = deepcopy(chess_board)
        execute_move(sim_board, move, you)
      
        val = self.minVal(sim_board, move, you, opp, a, b, i, start_time, move)

        if val > best_score:
          best_score = val
          best_move = move  

    self.turns += 1
    
    # time_taken = time.time() - start_time
    # print( time_taken, "seconds.")
    return best_move or random_move(chess_board, you)
    

  ###Generic MAXVAL Helper###
  def maxVal(self, board, move, you, opp, a, b, i, start_time, og_move):
    time_taken = time.time() - start_time

    if i == 0 or time_taken > 1.8:#cutoff condition
      return self.evalMove(board, move, you, opp)
    
    legal_moves = get_valid_moves(board, you)
    move_set = self.split_moves(legal_moves)

    for moves in move_set:
      for move in moves:
        
        sim_board = deepcopy(board)
        execute_move(sim_board, move, you)

        a = max(a, self.minVal(sim_board, move, you, opp, a, b, i - 1, start_time,og_move))
        
        if a >= b:
          return b
    return a
    
  ###Generic MINVAL Helper###
  def minVal(self, board, move, you, opp, a, b, i, start_time,og_move):
    time_taken = time.time() - start_time

    if i == 0 or time_taken > 1.8:
      return self.evalMove(board, move, you, opp)
    
    legal_moves = get_valid_moves(board, opp)
    move_set = self.split_moves(legal_moves)

    for moves in move_set:
      for move in moves:
        
        sim_board = deepcopy(board)
        execute_move(sim_board, move, opp)
      
        b = min(b, self.maxVal(sim_board, move, you, opp, a, b, i - 1, start_time,og_move))
        
        if a >= b:
          return a
    return b
  


  def evalMove(self, board, move, you, opp):
    if check_endgame(board)[0]:
      return 3550336 #big number (5th perfect number)
    
    btype = self.get_board_type(board)
    w = EVAL_WEIGHTS[btype] #gets multipliers based on board type
    
    player_count = np.count_nonzero(board == you)
    opp_count = np.count_nonzero(board == opp)
    score_diff = player_count - opp_count 
    gap_points = self.gap_points(board, move, w["gap_offender"], w["gap_defender"], you, opp)
    mobility_penalty = self.mobility(board, you, opp)
    move_bonus =  self.move_bonus(move, you) 
    opp_hole_bonus = self.opp_hole_bonus(board, move, you, opp)
   
    total_score = (w["score_diff"]     * score_diff + 
                   w["mobility"]       * mobility_penalty +
                   w["move_bonus"]     * move_bonus +
                   w["opp_hole_bonus"] * opp_hole_bonus +
                   2                   * gap_points 
                  
                  )
    
    return total_score
  

  ### Mobility function: difference between your valid moves and opponent's valid moves ###
  def mobility(self, board, you, opp):
    opp_moves = len(get_valid_moves(board, opp))
    you_moves = len(get_valid_moves(board, you))

    if self.get_board_type(board) == "wall": #added this condition after testing showed wall board needed to stop enemy mobility more
      return - opp_moves
    
    return you_moves - opp_moves
  

  ### Gap points function: rewards having pieces next to each other and penalizes leaving gaps ###
  def gap_points (self, board, move, offender_w, defender_w, you, opp):
    n = board.shape[0] 
    gap_penalty = 0
    gap_reward = 0
   
    at_row, at_col = move.get_src()
    dest_row, dest_col = move.get_dest()

    is_jump_move = abs(dest_col - at_col) == 2 or abs(dest_row - at_row) == 2
    for row in range(n):
      for col in range(n):
        around_you = (board[row][col] == you)
        jump = (row == at_row and col == at_col and board[at_row][at_col] == 0 and is_jump_move)

        if around_you or jump:
          for row_diff, col_diff in self.dir_coords:
            friend_pos = row + row_diff ,  col + col_diff
          
            if 0 <= friend_pos[0] < n and 0 <= friend_pos[1] < n:
              if self.get_board_type(board) == "wall": #on wall board, obstacles are taken more into account
                if board[friend_pos[0], friend_pos[1]] in [you, 3]:
                  if around_you:
                    gap_reward += 1 
                  if jump:
                    gap_penalty += 1
              else:
                if board[friend_pos[0], friend_pos[1]] == you :
                  if around_you:
                    gap_reward += 1
                  if jump:
                    gap_penalty += 1

    return gap_reward * offender_w - gap_penalty * defender_w
              
  ### Opponent hole bonus: rewards moves that place pieces into opponent gaps ###            
  def opp_hole_bonus(self, board, move, you, opp):
    n = board.shape[0] 
    dest_row, dest_col = move.get_dest()
    opp_hole_bonus = 0

    for row_diff, col_diff in self.dir_coords:
      opp_pos_row, opp_pos_col = dest_row + row_diff ,  dest_col + col_diff

      if 0 <=  opp_pos_row < n and 0 <= opp_pos_col< n:
        if board[opp_pos_row][opp_pos_col] == opp:
          opp_hole_bonus += 1
          
    return opp_hole_bonus


  ### Move bonus: slight penalty for jump moves to encourage duplicates ###
  def move_bonus(self, move, you):
    at_row, at_col = move.get_src()
    dest_row, dest_col = move.get_dest()

    if abs(dest_col - at_col) == 2 or abs(dest_row - at_row) == 2:
      return -5
    else:
      return 0

  ### Split moves into duplicate and jump moves, duplicates checked first ###
  def split_moves(self, moves):
    
    duplicate_moves = []
    jump_moves = []

    for move in moves:
      at_row, at_col = move.get_src()
      dest_row, dest_col = move.get_dest()
      if self.is_difficult:
        if abs(dest_row - at_row) == 2 or abs(dest_col - at_col) == 2:
          jump_moves.append(move)

        else:
          duplicate_moves.append(move)

      else: 
        if dest_row - at_row == 2 or dest_col - at_col == 2:
          jump_moves.append(move)
        else:
          duplicate_moves.append(move)

    move_set = [duplicate_moves, jump_moves]
    return move_set


  ### Determine board type based on obstacle positions ###
  def get_board_type(self, board):
    if (board[1, 1] == 3):
      # Board is big x 
      return "big_x"
    if (board[2, 3] == 3 and board[1, 3] != 3):
      # Board is plus1
      return "plus1"
    if (board[1, 3] == 3 and board[3, 0] != 3):
      # board is plus2
      return "plus2"
    if (board[2, 2] == 3):
      # board is point4
      return "point4"
    if (board[1, 2] == 3):
      # Board is the circle
      return "circle"
    if (board[0, 3] == 3):
      # the wall, we want to take over spots that are harder for the opponent to attack
      return "wall"

    if (board[0, 2] == 3):
      # Board is watch the sides 
      return "watch_sides"
    else:
      # Board is empty
      return "empty"    
    

        
 
        