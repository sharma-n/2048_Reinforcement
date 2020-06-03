import numpy as np

def new_game(gridsize=4):
    '''Start new 2048 game.
    :param gridsize: size of the grid for new game
    :return grid: (nxn) np array with 2 non-zero numbers inserted
    '''
    grid = np.zeros((gridsize, gridsize), dtype=np.int)
    grid = insert_new(grid)
    grid = insert_new(grid)
    return grid

def insert_new(grid):
    '''Add new tile in a random empty space
    :param grid: grid to insert tile in
    :return grid: grid with newly inserted tile
    '''
    xs, ys = np.nonzero(grid==0)
    if len(xs)==0: return grid
    add_id = np.random.randint(len(xs))
    if np.random.rand() < 0.9:
        grid[xs[add_id], ys[add_id]] = 2
    else:
        grid[xs[add_id], ys[add_id]] = 4
    return grid

def game_over(grid):
        '''Check if moves possible
        :param grid: game grid
        :return game_over: True if no more moves possible, False otherwise
        '''
        n = grid.shape[0]
        for i in range(n-1):
            if any(grid[i, :] == grid[i+1, :]) or any(grid[:, i]==grid[:, i+1]):
                return False
        if 0 in grid:
            return False
        return True

def flip(grid):
    '''Flip the grid along the columns
    '''
    return np.flip(grid, axis=1)

def transpose(grid):
    '''Change grid to its transpose
    '''
    return np.transpose(grid)

def cover_up(grid):
    '''Push all zeros to the right-hand side
    :return grid: game grid after cover up
    :return changed: True if any changes done on the grid, False otherwise
    '''
    valid_mask = grid!=0
    flipped_mask = valid_mask.sum(1,keepdims=1) > np.arange(grid.shape[1]-1,-1,-1)
    flipped_mask = flipped_mask[:,::-1]
    changed = np.any(grid[~flipped_mask]!=0)
    grid[flipped_mask] = grid[valid_mask]
    grid[~flipped_mask] = 0
    return grid, changed

def merge(grid):
    '''Combine tiles of same value along the rows
    :return grid: game grid after merging
    :return scoreChanged: True if any tiles merged, False otherwise
    :return score: Score after merging. 0 if scoreChanged is False
    '''
    score = 0
    n = grid.shape[0]
    for i in range(n-1):
        same_ids = np.where(grid[:, i]==grid[:, i+1])
        grid[same_ids, i] *= 2
        score += np.sum(grid[same_ids, i])
        grid[same_ids, i+1] = 0
    return grid, score!=0, score

def up(grid):
    '''Up movement
    :return grid: Game grid after movement
    :return done: True if any changes done to the grid
    :return score: Score collected in single movement
    '''
    grid = transpose(grid)
    grid, done = cover_up(grid)
    grid, haveScore, score = merge(grid)
    done = done or haveScore
    grid, _ = cover_up(grid)
    grid = transpose(grid)
    return grid, done, score

def down(grid):
    '''Down movement
    :return grid: Game grid after movement
    :return done: True if any changes done to the grid
    :return score: Score collected in single movement
    '''
    grid = transpose(grid)
    grid = flip(grid)
    grid, done = cover_up(grid)
    grid, scoreChanged, score = merge(grid)
    done = done or scoreChanged
    grid, _ = cover_up(grid)
    grid = flip(grid)
    grid = transpose(grid)
    return grid, done, score

def left(grid):
    '''Left movement
    :return grid: Game grid after movement
    :return done: True if any changes done to the grid
    :return score: Score collected in single movement
    '''
    grid, done = cover_up(grid)
    grid, scoreChanged, score = merge(grid)
    done = done or scoreChanged
    grid, _ = cover_up(grid)
    return grid, done, score

def right(grid):
    '''Right movement
    :return grid: Game grid after movement
    :return done: True if any changes done to the grid
    :return score: Score collected in single movement
    '''
    grid = flip(grid)
    grid, done = cover_up(grid)
    grid, scoreChanged, score = merge(grid)
    done = done or scoreChanged
    grid, _ = cover_up(grid)
    grid = flip(grid)
    return grid, done, score

def countEmptyCells(grid):
    '''Count the number of empty cells remaining on grid
    :return count: (int) number of empty cells
    '''
    return np.count_nonzero(grid==0)
