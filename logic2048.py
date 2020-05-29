import numpy as np

SIZE = 400
GRID_LEN = 4
GRID_PADDING = 10

BACKGROUND_COLOR_GAME = "#92877d"
BACKGROUND_COLOR_CELL_EMPTY = "#9e948a"

BACKGROUND_COLOR_DICT = {2: "#eee4da", 4: "#ede0c8", 8: "#f2b179",
                         16: "#f59563", 32: "#f67c5f", 64: "#f65e3b",
                         128: "#edcf72", 256: "#edcc61", 512: "#edc850",
                         1024: "#edc53f", 2048: "#edc22e",

                         4096: "#eee4da", 8192: "#edc22e", 16384: "#f2b179",
                         32768: "#f59563", 65536: "#f67c5f", }

CELL_COLOR_DICT = {2: "#776e65", 4: "#776e65", 8: "#f9f6f2", 16: "#f9f6f2",
                   32: "#f9f6f2", 64: "#f9f6f2", 128: "#f9f6f2",
                   256: "#f9f6f2", 512: "#f9f6f2", 1024: "#f9f6f2",
                   2048: "#f9f6f2",

                   4096: "#776e65", 8192: "#f9f6f2", 16384: "#776e65",
                   32768: "#776e65", 65536: "#f9f6f2", }

FONT = ("Verdana", 40, "bold")

KEY_UP_ALT = "\'\\uf700\'"
KEY_DOWN_ALT = "\'\\uf701\'"
KEY_LEFT_ALT = "\'\\uf702\'"
KEY_RIGHT_ALT = "\'\\uf703\'"

KEY_UP = "'w'"
KEY_DOWN = "'s'"
KEY_LEFT = "'a'"
KEY_RIGHT = "'d'"
KEY_BACK = "'b'"

class Logic2048():
    def __init__(self, gridsize=4):
        self.n = gridsize
        self.grid = np.zeros((gridsize, gridsize))

    def insert_new(self):
        '''Add new tile in a random empty space
        '''
        xs, ys = np.nonzero(self.grid==0)
        if len(xs)==0: return
        add_id = np.random.randint(len(xs))
        if np.random.rand() < 0.9:
            self.grid[xs[add_id], ys[add_id]] = 2
        else:
            self.grid[xs[add_id], ys[add_id]] = 4
        return
    
    def game_over(self):
        '''Check if moves possible
        :return game_over: True if no more moves possible, False otherwise
        '''
        for i in range(self.n-1):
            if any(self.grid[i, :] == self.grid[i+1, :]) or any(self.grid[:, i]==self.grid[:, i+1]):
                return False
        if 0 in self.grid:
            return False
        return True

    def flip(self):
        '''Flip the grid along the columns
        '''
        self.grid = np.flip(self.grid, axis=1)
        return

    def transpose(self):
        '''Change grid to its transpose
        '''
        self.grid = np.transpose(self.grid)
        return

    def cover_up(self):
        '''Push all zeros to the right-hand side
        :return changed: True if any changes done on the grid, False otherwise
        '''
        valid_mask = self.grid!=0
        flipped_mask = valid_mask.sum(1,keepdims=1) > np.arange(self.grid.shape[1]-1,-1,-1)
        flipped_mask = flipped_mask[:,::-1]
        changed = np.any(self.grid[~flipped_mask]!=0)
        self.grid[flipped_mask] = self.grid[valid_mask]
        self.grid[~flipped_mask] = 0
        return changed

    def merge(self):
        '''Combine tiles of same value along the rows
        :return scoreChanged: True if any tiles merged, False otherwise
        :return score: Score after merging. 0 if scoreChanged is False
        '''
        score = 0
        for i in range(self.n-1):
            same_ids = np.where(self.grid[:, i]==self.grid[:, i+1])
            self.grid[same_ids, i] *= 2
            score += np.sum(self.grid[same_ids, i])
            self.grid[same_ids, i+1] = 0
        return score!=0, score

    def up(self):
        '''Up movement
        :return done: True if any changes done to the grid
        :return score: Score collected in single movement
        '''
        self.transpose()
        done = self.cover_up()
        haveScore, score = self.merge()
        done = done or haveScore
        _ = self.cover_up()
        self.transpose()
        return done, score

    def down(self):
        '''Down movement
        :return done: True if any changes done to the grid
        :return score: Score collected in single movement
        '''
        self.transpose()
        self.flip()
        done = self.cover_up()
        scoreChanged, score = self.merge()
        done = done or scoreChanged
        _ = self.cover_up()
        self.flip()
        self.transpose()
        return done, score

    def left(self):
        '''Left movement
        :return done: True if any changes done to the grid
        :return score: Score collected in single movement
        '''
        done = self.cover_up()
        scoreChanged, score = self.merge()
        done = done or scoreChanged
        _ = self.cover_up()
        return done, score

    def right(self):
        '''Right movement
        :return done: True if any changes done to the grid
        :return score: Score collected in single movement
        '''
        self.flip()
        done = self.cover_up()
        scoreChanged, score = self.merge()
        done = done or scoreChanged
        _ = self.cover_up()
        self.flip()
        return done, score

    def countEmptyCells(self):
        '''Count the number of empty cells remaining on grid
        :return count: (int) number of empty cells
        '''
        return np.count_nonzero(self.grid==0)

    def change_values(self):
        raise NotImplementedError