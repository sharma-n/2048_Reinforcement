from logic2048 import *
from simpleRL import Simple_RLAgent
from cfgs.config import cfg, cfg_from_yaml_file
import tensorflow as tf
import numpy as np
from argparse import ArgumentParser
from matplotlib import pyplot as plt
import tqdm

def arg_parse():
    ap = ArgumentParser(description='Run performance tests on RL agents playing 2048')
    ap.add_argument('--cfg', required=True, help='Path to config file')
    ap.add_argument('--n_runs', type=int, default=10000, help='Number of games to be played by each agent')
    ap.add_argument('--viz', action='store_true', help='If you want to visualize one game run')
    args = ap.parse_args()
    cfg_from_yaml_file(args.cfg, cfg)
    return args

def run_game(model, n_runs):
    controls = {0:up, 1:left, 2:right, 3:down}
    n_moves, max_tiles, scores = [], [], []
    pbar = tqdm.tqdm(total=n_runs, desc='Game runs', dynamic_ncols=True)
    for _ in range(n_runs):
        grid = new_game(cfg.GAME_CONFIG.GRID_LEN)
        ep_score, ep_move = 0, 0
        finish = False
        while True:
            state = model.power_grid(grid)
            control_scores = model(np.expand_dims(state, axis=0))
            control_buttons = np.flip(np.argsort(control_scores),axis=1)
            for move in control_buttons[0]:
                prev_state = grid.copy()
                temp_grid, changed, move_score = controls[move](prev_state)
                if not changed: #illegal move
                    continue
                else:
                    break

            ep_score += move_score
            ep_move += 1
            grid = temp_grid.copy()
            grid = insert_new(grid)
            if game_over(grid):
                break
        scores.append(ep_score)
        n_moves.append(ep_move)
        max_tiles.append(np.max(grid))
        pbar.update()
        pbar.set_postfix({'Avg. score':np.average(scores), 'Avg. max tile': np.average(max_tiles)})
    
    return n_moves, max_tiles, scores

def plot_results(n_moves, max_tiles, scores):
    plt.figure()
    plt.hist(n_moves)
    plt.title('Distribution of number of moves played')
    plt.xlabel('Number of moves')
    plt.ylabel('Number of games')
    plt.show()

    plt.figure()
    plt.hist(max_tiles)
    plt.title('Distribution of max tile achieved')
    plt.xlabel('Maximum tile value')
    plt.ylabel('Number of games')
    plt.show()

    plt.figure()
    plt.hist(scores)
    plt.title('Distribution of final score achieved')
    plt.xlabel('Final Score')
    plt.ylabel('Number of games')
    plt.show()

def main(args):
    model = Simple_RLAgent(train=False)
    latest = tf.train.latest_checkpoint(cfg.MODEL.CHECKPOINT_DIR)
    model.load_weights(latest)
    print('[INFO    ] Model loaded from: ', latest)
    if args.viz:
        from gui import Frame2048
        from tkinter import Tk
        root = Tk()
        grid = Frame2048(human_player=False, model=model)
        root.mainloop()
        return

    n_moves, max_tiles, scores = run_game(model, args.n_runs)
    plot_results(n_moves, max_tiles, scores)

if __name__=='__main__':
    main(arg_parse())