# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, execute_move, check_endgame, get_valid_moves

@register_agent("student_agent")
class StudentAgent(Agent):
  """
  A class for your implementation. Feel free to use this class to
  add any helper functionalities needed for your agent.
  """
  boards = []
  def __init__(self):
    super(StudentAgent, self).__init__()
    self.name = "StudentAgent"

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
    
    move_set = self.split_moves(legal_moves)

    a = float('-inf')
    b =  float('inf')
    best_score = float('-inf')
    best_move = None
    i = 3
    n = chess_board.shape[0] 
    at_row, at_col = 0, 0
    dest_row, dest_col = 0, 0
    self.boards = []
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
    print(best_score)
    # for move in legal_moves:
      
    #   pieces = np.count_nonzero(chess_board)
    #   enemy_pieces = np.count_nonzero(chess_board == opp)
    #   your_pieces = np.count_nonzero(chess_board == you)
    #   at_row, at_col = move.get_src()
    #   dest_row, dest_col = move.get_dest()
      
    #   if not(dest_row - at_row == 2 or dest_col - at_col == 2):
        
    #     sim_board = deepcopy(chess_board)

    #     execute_move(sim_board, move, you)
        
        
    #     val = self.minVal(sim_board, move, you, opp, a, b, i, start_time)

    #     if val > best_score:
    #       best_score = val
    #       best_move = move  
  
    # for move in legal_moves:
      
    #   pieces = np.count_nonzero(chess_board)
    #   enemy_pieces = np.count_nonzero(chess_board == opp)
    #   your_pieces = np.count_nonzero(chess_board == you)
    #   at_row, at_col = move.get_src()
    #   dest_row, dest_col = move.get_dest()
      
    #   if (dest_row - at_row == 2 or dest_col - at_col == 2):
        
    #     sim_board = deepcopy(chess_board)

    #     execute_move(sim_board, move, you)
        
        
    #     val2 = self.minVal(sim_board, move, you, opp, a, b, i, start_time)

    #     if val2 > best_score:
    #       best_score = val2
    #       best_move = move  


    
    
    time_taken = time.time() - start_time

    print("My AI's turn took ", time_taken, "seconds.")
    
    return best_move or random_move(chess_board, you)
    
    
  
  

  def maxVal(self, board, move, you, opp, a, b, i, start_time,og_move):
    time_taken = time.time() - start_time

    if i == 0 or time_taken > 1.85:
      # sim_board = deepcopy(board)
      # execute_move(sim_board, move, opp)
      return self.evalMove(board, move, you, opp, i, og_move)
    legal_moves = get_valid_moves(board, you)
    move_set = self.split_moves(legal_moves)

    for moves in move_set:
      for move in moves:
        
        sim_board = deepcopy(board)
      
        execute_move(sim_board, move, you)
        if self.board_checker(sim_board):
          continue
        else:
          self.boards.append(sim_board)
        a = max(a, self.minVal(sim_board, move, you, opp, a, b, i - 1, start_time,og_move))
        
        if a >= b:
          return b
       
    return a
    
 
  def minVal(self, board, move, you, opp, a, b, i, start_time,og_move):
    time_taken = time.time() - start_time

    if i == 0 or time_taken > 1.85:
      # sim_board = deepcopy(board)
      # execute_move(sim_board, move, you)
      return self.evalMove(board, move, you, opp, i, og_move)
    
    legal_moves = get_valid_moves(board, opp)
    move_set = self.split_moves(legal_moves)

    for moves in move_set:
      for move in moves:
        
        sim_board = deepcopy(board)
        execute_move(sim_board, move, opp)
        
        if self.board_checker(sim_board):
          continue
        else:
          self.boards.append(sim_board)
        b = min(b, self.maxVal(sim_board, move, you, opp, a, b, i - 1, start_time,og_move))
        
        if a >= b:
          return a
        
    return b
  
  def evalMove(self, board, move, you, opp, i, og_move):
    finishing_move = self.last_move_win(board, you, opp)
    player_count = np.count_nonzero(board == you)
    opp_count = np.count_nonzero(board == opp)
    score_diff = player_count - opp_count
    corner_bonus = 0
    n = board.shape[0]
    
    mobility_penalty = self.mobility(board, you, opp) * 2
    gap_defender = self.gap_defender(board, og_move, you, opp) * 20
    gap_offender = self.gap_offender(board, og_move, you, opp) * 10

    # gap_penalty = 0
    edge_bonus = self.edge_bonus(board, you) * 20 * 0
    center_bonus = self.center_bonus(board, you) * 2 * 0

    move_bonus = self.move_bonus(move, you) * 5 * 0

    total_score = (score_diff + 
                   gap_offender + 
                   gap_defender + 
                   corner_bonus + 
                   mobility_penalty +
                   edge_bonus +  
                   finishing_move + 
                   center_bonus + 
                   move_bonus
                   )
    
    return total_score
  
  def evalMove1(self, board, move, you, opp, i, og_move):
    finishing_move = self.last_move_win(board, you, opp)
    player_count = np.count_nonzero(board == you)
    opp_count = np.count_nonzero(board == opp)
    score_diff = player_count - opp_count
    return score_diff + finishing_move

  def last_move_win(self, board, you, opp):
    if check_endgame(board)[0]:
      you_pts = np.count_nonzero(board == you)
      opp_pts = np.count_nonzero(board == opp)
      return 33550336 * (you_pts - opp_pts)
    return 0

  def mobility(self, board, you, opp):
    opp_moves = len(get_valid_moves(board, opp))
    you_moves = len(get_valid_moves(board, you))
    return -opp_moves
  
  def gap_defender(self, board, move, you, opp):
    n = board.shape[0] 
    gap_penalty = 0
   
    at_row, at_col = move.get_src()
    dest_row, dest_col = move.get_dest()
    dir_coords = [
        (-1,-1), (-1, 0), (-1, 1),
        ( 0,-1),          ( 0, 1),
        ( 1,-1), ( 1, 0), ( 1, 1)
    ]
 
    if (dest_col - at_col) == 2 or (dest_row - at_row) == 2:
      for row in range(n):
        for col in range(n):
          if board[row][col] == 0:
            opp_close_by = 0

            for row_diff, col_diff in dir_coords:
              opp_pos = row + row_diff ,  col + col_diff
            
              if 0 <= opp_pos[0] < n and 0 <= opp_pos[1] < n:
                if board[opp_pos[0], opp_pos[1]] == you:
                  opp_close_by += 1
            if opp_close_by >= 5:
              gap_penalty -= opp_close_by
            

    return gap_penalty

  def gap_offender(self, board, move, you, opp):
    n = board.shape[0] 
    gap_penalty = 0
   
    at_row, at_col = move.get_src()
    dest_row, dest_col = move.get_dest()
    dir_coords = [
        (-1,-1), (-1, 0), (-1, 1),
        ( 0,-1),          ( 0, 1),
        ( 1,-1), ( 1, 0), ( 1, 1)
    ]
        
    for row in range(n):
      for col in range(n):
        if board[row][col] == you:
          opp_close_by = 0

          for row_diff, col_diff in dir_coords:
            opp_pos = row + row_diff ,  col + col_diff
           
            if 0 <= opp_pos[0] < n and 0 <= opp_pos[1] < n:
              if board[opp_pos[0], opp_pos[1]] == you:
                opp_close_by += 1
          
          gap_penalty += opp_close_by 

    return gap_penalty
    # if board[dest_row][dest_col] == 0:
    #   if n-2 > dest_row >= 1 and n-2 > dest_col >= 1 :
    #       if board[dest_row - 1][dest_col] == opp:
    #         count += 1
    #       if board[dest_row + 1][dest_col] == opp:
    #         count += 1
    #       if board[dest_row][dest_col - 1] == opp:
    #         count += 1
    #       if board[dest_row][dest_col - 1] == opp:
    #         count += 1

    #       if board[dest_row - 1][dest_col - 1] == opp:
    #         count += 1
    #       if board[dest_row + 1][dest_col - 1] == opp:
    #         count += 1
    #       if board[dest_row - 1][dest_col + 1] == opp:
    #         count += 1
    #       if board[dest_row + 1][dest_col + 1] == opp:
    #         count += 1
      
    return count
  
  def edge_bonus(self, board, you):
    n = board.shape[0] 
    edges = []
    edges.append((0,0))
    edges.append((0,1))
    edges.append((1,0))
    edges.append((1,1))
    edges.append((n-1,n-1))
    edges.append((n-2,n-1))
    edges.append((n-1,n-2))
    edges.append((n-2,n-2))
    edges.append((0,n-1))
    edges.append((0,n-2))
    edges.append((1,n-1))
    edges.append((1,n-2))
    edges.append((n-1,0))
    edges.append((n-2,0))
    edges.append((n-1,1))
    edges.append((n-2,1))

    edge_bonus = sum(1 for (i, j) in edges if board[i, j] == you)
    return edge_bonus

  def center_bonus(self, board, you):
    return 0
    n = board.shape[0] 
    center = [
      (n//2 - 1, n//2 -1), (n//2 - 1, n//2), (n//2 - 1, n//2 + 1), 
      (n//2, n//2 - 1),        (n//2, n//2), (n//2,n//2 + 1),
      (n//2 + 1, n//2 -1), (n//2 + 1, n//2), (n//2 + 1, n//2 + 1)
    ]
    center_bonus = sum(1 for (i, j) in center if board[i, j] == you)
    return center_bonus
  
  def move_bonus(self, move, you):
    at_row, at_col = move.get_src()
    dest_row, dest_col = move.get_dest()

    if (dest_col - at_col) == 2 or (dest_row - at_row) == 2:
      return -5
    else:
      return 0

  def split_moves(self, moves):
    
    duplicate_moves = []
    jump_moves = []

    for move in moves:
      at_row, at_col = move.get_src()
      dest_row, dest_col = move.get_dest()

      if dest_row - at_row == 2 or dest_col - at_col == 2:
        jump_moves.append(move)
      else:
        duplicate_moves.append(move)

    move_set = [duplicate_moves, jump_moves]
    return move_set



  def board_checker(self, board):
    n = board.shape[0]
    
    for old_board in self.boards:
      return False
      if np.array_equal(old_board, board):
        return True
   
        

      # count = 0
      # flag = False
      # for row in range(n):
      #   for col in range(n):
      #     if not old_board[row][col] == board[row][col]:
      #       flag = True
      #     else:
      #       count += 1

      #     if flag:
      #       break
      #   if flag:
      #     break
      # if count == n**2:
      #   print("DUPLICATE MOVE FOUND")
      #   return True
      
    return False

  def evalMove2(self, board, move, you, opp, i):
    if check_endgame(board):
      return 5000
    player_count = np.count_nonzero(board == you)
    opp_count = np.count_nonzero(board == opp)
    score_diff = player_count - opp_count
    at_row, at_col = move.get_src()
    dest_row, dest_col = move.get_dest()
    jmp_move = 0
    n = board.shape[0] 
    zeros = n^2 - np.count_nonzero(board)
    
    

    # mvs = get_valid_moves(board, you)
    # for mv in mvs :
    #   at_row2, at_col2 = mv.get_src()
    #   dest_row2, dest_col2 = mv.get_dest()
    #   if abs(dest_row2 - at_row2) == 2 or abs(dest_col2 - at_col2) == 2:
    #     jmp_moves += 1
   

    # corner control bonus
   
   

    if abs(dest_row - at_row) == 2 or abs(dest_col - at_col) == 2:
      zeros = n^2 - np.count_nonzero(board)
      print(player_count + opp_count == np.count_nonzero(board))
      if zeros >= (n**2) / 4:
        jmp_move = - 50
    # # for x in range(n):
    # #   for y in range(n):
    # #     if board[x][y] == 0:
    # #       i = 0
    # #       if  x - 1 >= 0:


    # #       adj = 
    # #   return 0
    corners = [(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)]
    
    corner_bonus = sum(1 for (i, j) in corners if board[i, j] == you) * 5
    # edges = []
    # for x in range(n):
    #   edges.append((0,x))
    #   edges.append((x,0))
    #   edges.append((n-1,x))
    #   edges.append((x,n-1))
    # edge_bonus = sum(1 for (i, j) in edges if board[i, j] == you)
    # penalize opponent mobility
    opp_moves = len(get_valid_moves(board, opp))
    mobility_penalty = -opp_moves

    return score_diff + jmp_move + mobility_penalty

        
