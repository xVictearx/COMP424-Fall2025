# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, execute_move, check_endgame, get_valid_moves


EVAL_WEIGHTS = {
    # board[1,1] == 3
    "big_x": {
        "score_diff":   1,   # material still matters
        "gap_offender": 25,   # small reward for blobs
        "gap_defender": 35,   # mild penalty for bad jumps leaving holes
        "move_bonus":   0,   # jumps slightly punished (your move_bonus is 0 or -5)
        #"board_bonus":  0.0,   # no special positional mask
        "mobility": 5,   # open so mobility important
        "obst_bonus":  0,
        "obstacle": []
    },

    # board[2,3] == 3 and board[1,3] != 3
    "plus1": {
        "score_diff":   1,
        "gap_offender": 35,   # blobs around cross arms are nice
        "gap_defender": 30,   # avoid leaving big 1-holes near the plus
        "move_bonus":   40,   # jumps are somewhat risky
        #"board_bonus":  0.8,   # if you later add a plus1 bonus grid
        "mobility": 5,   # open so mobility important
        "obst_bonus":  0,
        "obstacle": []
    },

    # board[1,3] == 3
    "plus2": {
        "score_diff":   1,
        "gap_offender": 40,   # like plus1 but a bit stronger clustering
        "gap_defender": 40,
        "move_bonus":   5,
        #"board_bonus":  1.0,
        "mobility": 0,   # open so mobility important
        "obst_bonus":  0,
        "obstacle": []
    },

    # board[2,2] == 3
    "point4": {
        "score_diff":   1,
        "gap_offender": 30,   # islands-ish: safe blobs good
        "gap_defender": 35,   # really punish dangerous gaps
        "move_bonus":   60,
        #"board_bonus":  1.2,   # central-ish control good
        "mobility": 10,   # open so mobility important
        "obst_bonus":  10,
        "obstacle": []
    },

    # board[1,2] == 3
    "circle": {
        "score_diff":   1,
        "gap_offender": 35,   # ring control
        "gap_defender": 35,
        "move_bonus":   20,
        #"board_bonus":  1.0,   # if you have a ring bonus mask later
        "mobility": 5,   # open so mobility important
        "obst_bonus":  0,
        "obstacle": []
        
    },

    # board[0,3] == 3
    "wall": {
        "score_diff":   1,
        "gap_offender": 5,   # big safe blobs behind wall are great
        "gap_defender": 20,   # strongly avoid 1-hole disasters
        "move_bonus":   0,   # jumps more dangerous here
        #"board_bonus":  2.0,   # chokepoints near the wall are super valuable
        "mobility": 100,   
        "obst_bonus":  10,
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
        "score_diff":   1,
        "gap_offender": 40,   # side clusters good
        "gap_defender": 40,
        "move_bonus":   60,
        #"board_bonus":  1.5,   # strongly prefer good side squares
        "mobility": 5,   # open so mobility important
        "obst_bonus":  0,
        "obstacle": []
    },

    # else: empty / no obstacles
    "empty": {
        "score_diff":   1,
        "gap_offender": 15,   # moderate value for clustering
        "gap_defender": 20,   # avoid terrible gaps but not overkill
        "move_bonus":   35,   # your original: jumps punished by full move_bonus
        #"board_bonus":  0.5,   # mild center-ish preference if you use it
        "mobility": 5,   # open so mobility important
        "obst_bonus":  0,
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

  def step(self, chess_board, you, opp):
    """
      You should return a tuple (r,c), where (r,c) is the position where your agent
    wants to place the next disc. Use functions in helpers to determine valid moves
    and more helpful tools.

    Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
    """
    
    start_time = time.time()
    
    legal_moves = get_valid_moves(chess_board, you)
    
    if not legal_moves:
      return None  # No valid moves available, pass turn
    
    
    
    a = float('-inf')
    b =  float('inf')
    best_score = float('-inf')
    best_move = None
    i = 3
    n = chess_board.shape[0] 
    at_row, at_col = 0, 0
    dest_row, dest_col = 0, 0
    pieces = np.count_nonzero(chess_board != 0)
    self.boards = []
    board_type = self.get_board_type(chess_board)
    move_set = []
    
    # if self.turns < 3 and (boardt_type == "wall" or boardt_type == "watch_sides" or boardt_type == "circle"):
    if self.turns < 3 or (board_type in ["point4", "plus2"] and pieces > n**2 - n):
      # if boardt_type == "wall" :
      #   self.turns = 2
      # self.is_difficult = True
      move_set = self.split_moves(legal_moves)
      
      move_set = [move_set[0]]
    else:
      move_set = self.split_moves(legal_moves)

      # print(move_set)
      
    for moves in move_set:
      for move in moves:
        
        pieces = np.count_nonzero(chess_board)
        enemy_pieces = np.count_nonzero(chess_board == opp)
        your_pieces = np.count_nonzero(chess_board == you)
        at_row, at_col = move.get_src()
        dest_row, dest_col = move.get_dest()
      
       
          
        sim_board = deepcopy(chess_board)

        execute_move(sim_board, move, you)
        
        
        val = self.minVal(sim_board, move, you, opp, a, b, i, start_time, move)

        if val > best_score:
          best_score = val
          best_move = move  

    self.turns += 1
    # print(best_score)
 
    
    
    time_taken = time.time() - start_time

    print("My AI's turn took ", time_taken, "seconds.")
    
    return best_move or random_move(chess_board, you)
    
    
  
  

  def maxVal(self, board, move, you, opp, a, b, i, start_time,og_move):
    time_taken = time.time() - start_time

    if i == 0 or time_taken > 1.8:
      # sim_board = deepcopy(board)
      # execute_move(sim_board, move, opp)
      return self.evalMove(board, move, you, opp, i, og_move)
    legal_moves = get_valid_moves(board, you)
    move_set = self.split_moves(legal_moves)

    for moves in move_set:
      for move in moves:
        
        sim_board = deepcopy(board)
      
        execute_move(sim_board, move, you)
        
        # self.boards.append(sim_board)
        a = max(a, self.minVal(sim_board, move, you, opp, a, b, i - 1, start_time,og_move))
        
        if a >= b:
          return b
       
    return a
    
 
  def minVal(self, board, move, you, opp, a, b, i, start_time,og_move):
    time_taken = time.time() - start_time

    if i == 0 or time_taken > 1.8:
      # sim_board = deepcopy(board)
      # execute_move(sim_board, move, you)
      return self.evalMove(board, move, you, opp, i, og_move)
    
    legal_moves = get_valid_moves(board, opp)
    move_set = self.split_moves(legal_moves)

    for moves in move_set:
      for move in moves:
        
        sim_board = deepcopy(board)
        execute_move(sim_board, move, opp)
        
        
        # self.boards.append(sim_board)
        b = min(b, self.maxVal(sim_board, move, you, opp, a, b, i - 1, start_time,og_move))
        
        if a >= b:
          return a
        
    return b
  
  def evalMove(self, board, move, you, opp, i, og_move):
    btype = self.get_board_type(board)
    
    finishing_move = self.last_move_win(board, you, opp)
    if finishing_move == 33550336:
      return finishing_move
    player_count = np.count_nonzero(board == you)
    opp_count = np.count_nonzero(board == opp)
    score_diff = player_count - opp_count
    corner_bonus = 0
    n = board.shape[0]
    # board_bonus = self.find_bonus_board(board, move, you, opp)
    
    mobility_penalty = self.mobility(board, you, opp)
    gap_defender = self.gap_defender(board, og_move, you, opp) 
    gap_offender = self.gap_offender(board, og_move, you, opp)
    # gap_penalty = 0
    
    
    center_bonus =  2 * 0 #* self.center_bonus(board, you) 
    corner_bonus = 0
    # if self.get_board_type(board) == "plus2":
    #   corner_bonus =  50 * self.edge_bonus(board, you) 
    # if self.get_board_type(board) == "wall":
    #   center_bonus = 200 * self.center_bonus(board, you)
    move_bonus =  self.move_bonus(move, you) 
    #safe_territory_bonus = 10 * self.far_from_opp(board, move, you, opp)

    near_wall_bonus = self.near_wall_bonus(board, you)

   
    w = EVAL_WEIGHTS[btype]
   
    total_score = (w["score_diff"] * score_diff + 
                   w["gap_offender"] * gap_offender + 
                   w["gap_defender"] * gap_defender + 
                   w["mobility"] * mobility_penalty +
                   w["move_bonus"] * move_bonus +
                   w["obst_bonus"] * near_wall_bonus +
                   
                   finishing_move + 
                   center_bonus + 
                   corner_bonus

                   # 0 * board_bonus 
                   #safe_territory_bonus 
                   )
    
    return total_score
  

  def last_move_win(self, board, you, opp):
    if check_endgame(board)[0]:
      you_pts = np.count_nonzero(board == you)
      opp_pts = np.count_nonzero(board == opp)
      return 33550336 
    return 0

  def mobility(self, board, you, opp):
    opp_moves = len(get_valid_moves(board, opp))
    you_moves = len(get_valid_moves(board, you))
    if self.get_board_type(board) == "wall":
      return - opp_moves
    return you_moves-opp_moves
  
  def gap_defender(self, board, move, you, opp):
    n = board.shape[0] 
    gap_penalty = 0
    i = 3
    if self.get_board_type(board) == "plus2":
      i = -2
    at_row, at_col = move.get_src()
    dest_row, dest_col = move.get_dest()
    dir_coords = [
        (-1,-1), (-1, 0), (-1, 1),
        ( 0,-1),          ( 0, 1),
        ( 1,-1), ( 1, 0), ( 1, 1)
    ]
    for row_diff, col_diff in dir_coords:
      pos = at_row + row_diff ,  at_col + col_diff
     
      if 0 <= pos[0] < n and 0 <= pos[1] < n:
        if board[pos[0]][pos[1]] in [you, i]:
              gap_penalty -= 1
        elif board[pos[0]][pos[1]] == 0:
              gap_penalty += 0
        
    
    # if abs(dest_col - at_col) == 2 or abs(dest_row - at_row) == 2:
    #   if board[at_row][at_col] == 0:
    #         opp_close_by = 0

    #         for row_diff, col_diff in dir_coords:
    #           opp_pos = at_row + row_diff ,  at_col + col_diff
              
    #           if 0 <= opp_pos[0] < n and 0 <= opp_pos[1] < n:
    #              if board[opp_pos[0], opp_pos[1]] == you or board[opp_pos[0], opp_pos[1]] == i:
    #                 opp_close_by += 1
                
    #         if opp_close_by >= 5:
    #           gap_penalty -= opp_close_by
  
            

    return gap_penalty

  def gap_offender(self, board, move, you, opp):
    n = board.shape[0] 
    gap_penalty = 0
    i = 3
    if self.get_board_type(board) == "plus2":
      i = -2
    at_row, at_col = move.get_src()
    dest_row, dest_col = move.get_dest()
    dir_coords = [
        (-1,-1), (-1, 0), (-1, 1),
        ( 0,-1),          ( 0, 1),
        ( 1,-1), ( 1, 0), ( 1, 1)
    ]
    for row_diff, col_diff in dir_coords:
      pos = dest_row + row_diff ,  dest_col + col_diff
     
      if 0 <= pos[0] < n and 0 <= pos[1] < n:
        if board[pos[0]][pos[1]] in [you, i]:
              gap_penalty += 1    
              
    # for row in range(n):
    #   for col in range(n):
    #     if board[row][col] == you:
    #       opp_close_by = 0

    #       for row_diff, col_diff in dir_coords:
    #         opp_pos = row + row_diff ,  col + col_diff
           
    #         if 0 <= opp_pos[0] < n and 0 <= opp_pos[1] < n:
    #           if board[opp_pos[0], opp_pos[1]] == you or board[opp_pos[0], opp_pos[1]] == i:
    #                 opp_close_by += 1
          
    #       gap_penalty += opp_close_by 

    return gap_penalty

  
  def edge_bonus(self, board, you):
    n = board.shape[0] 
    edges = []
    # edges.append((0,0))
    edges.append((0,1))
    edges.append((1,0))
    # edges.append((1,1))
    # edges.append((n-1,n-1))
    edges.append((n-2,n-1))
    edges.append((n-1,n-2))
    # edges.append((n-2,n-2))
    # edges.append((0,n-1))
    edges.append((0,n-2))
    edges.append((1,n-1))
    # edges.append((1,n-2))
    # edges.append((n-1,0))
    edges.append((n-2,0))
    edges.append((n-1,1))
    # edges.append((n-2,1))

    edge_bonus = sum(1 for (i, j) in edges if board[i, j] == you)
    return edge_bonus

  def center_bonus(self, board, you):
    
    n = board.shape[0] 
    center = [
      (n//2 - 1, n//2 -1), (n//2 - 1, n//2), (n//2 - 1, n//2 + 1), 
      (n//2, n//2 - 1),        (n//2, n//2), (n//2,n//2 + 1),
      (n//2 + 1, n//2 -1), (n//2 + 1, n//2), (n//2 + 1, n//2 + 1)
    ]
    center = [(n//2, n//2)]
    center_bonus = sum(1 for (i, j) in center if board[i, j] == you)
    return center_bonus
  
  def move_bonus(self, move, you):
    at_row, at_col = move.get_src()
    dest_row, dest_col = move.get_dest()

    if abs(dest_col - at_col) == 2 or abs(dest_row - at_row) == 2:
      return -5
    else:
      return 0

  def split_moves(self, moves):
    
    duplicate_moves = []
    jump_moves = []

    for move in moves:
      at_row, at_col = move.get_src()
      dest_row, dest_col = move.get_dest()
      
      if abs(dest_row - at_row) == 2 or abs(dest_col - at_col) == 2:
        jump_moves.append(move)

      else:
        duplicate_moves.append(move)

     

    move_set = [duplicate_moves, jump_moves]
    return move_set

  
  def get_board_type(self, board):
    if (board[1, 1] == 3):
      # Board is big x 
      return "big_x"
    if (board[2,3] == 3 and board[1, 3] != 3):
      # Board is plus1
      return "plus1"
    if (board[1, 3] == 3 and board[3,0] != 3):
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
    

    #bonus_board = [
    #    [ 0,  0,  10,  0,  10,  0,  0],
    #    [ 0,  0,  10,  0,  10,  0,  0],
    #    [ 10,  10, 10,  0, 10,  10,  10],
    #    [ 0,  0,  0,  10,  0,  0,  0],
    #    [ 10,  10, 10,  0, 10,  10,  10],
    #    [ 0,  0,  10,  0,  10,  0,  0],
    #    [ 0,  0,  10,  0,  10,  0,  0]
    #  ]
        
  def near_wall_bonus(self, board, you):
    board_type = self.get_board_type(board)
    w = EVAL_WEIGHTS[board_type]      
    near_wall_bonus = sum(1 for (i, j) in w["obstacle"] if board[i, j] == you)

    return near_wall_bonus
        