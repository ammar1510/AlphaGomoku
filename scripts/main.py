from env import GomokuEnv

env = GomokuEnv(board_size=15,num_envs=5)

env.reset()
env.step([[0,1],[1,2],[2,3],[3,4],[4,5]])
boards,current_player,game_over = env.get_state()
print(boards[1])
print("working in env")
