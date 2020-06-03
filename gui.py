from tkinter import Frame, Label, CENTER
from cfgs.config import cfg, cfg_from_yaml_file
from logic2048 import *

class Frame2048(Frame):
    def __init__(self, human_player=True, model=None):
        c = cfg.GAME_CONFIG
        self.humanPlayer = human_player
        if not human_player and model is None:
            raise Exception('human_player is false. Provide AI model to run')
        self.score = 0
        self.gamegrid = None
        Frame.__init__(self)
        self.grid()
        self.master.title('2048')
        self.grid_cells = []    # GUI elements
        self.init_grid()
        self.init_matrix()
        self.update_grid_cells()

        if human_player:
            
            self.master.bind("<Key>", self.key_down)

            self.commands = {c.KEY.UP: up, c.KEY.DOWN: down,
                        c.KEY.LEFT: left, c.KEY.RIGHT: right,
                        c.KEY.UP_ALT: up, c.KEY.DOWN_ALT: down,
                        c.KEY.LEFT_ALT: left, c.KEY.RIGHT_ALT: right}
            self.mainloop()
        else:
            self.wait_visibility()
            self.model = model
            self.commands = {0:up, 1:left, 2:right, 3:down}
            self.after(10, self.make_move)

    def init_grid(self):
        c = cfg.GAME_CONFIG
        background = Frame(self, bg=c.BACKGROUND_COLOR_GAME,
                           width=c.SIZE, height=c.SIZE)
        background.grid()

        for i in range(c.GRID_LEN):
            grid_row = []
            for j in range(c.GRID_LEN):
                cell = Frame(background, bg=c.BACKGROUND_COLOR_CELL_EMPTY,
                             width=c.SIZE / c.GRID_LEN,
                             height=c.SIZE / c.GRID_LEN)
                cell.grid(row=i, column=j, padx=c.GRID_PADDING,
                          pady=c.GRID_PADDING)
                t = Label(master=cell, text="",
                          bg=c.BACKGROUND_COLOR_CELL_EMPTY,
                          justify=CENTER, font=(c.FONT.STYLE, c.FONT.SIZE, c.FONT.EMPHASIS), width=5, height=2)
                t.grid()
                grid_row.append(t)

            self.grid_cells.append(grid_row)
        return
    
    def init_matrix(self):
        self.gamegrid = new_game(cfg.GAME_CONFIG.GRID_LEN)
        if self.humanPlayer:    self.history_matrices = list()

    def update_grid_cells(self):
        if self.gamegrid is None:
            self.update_idletasks()
            return
        c = cfg.GAME_CONFIG
        for i in range(c.GRID_LEN):
            for j in range(c.GRID_LEN):
                new_number = self.gamegrid[i][j]
                if new_number == 0:
                    self.grid_cells[i][j].configure(
                        text="", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                else:
                    self.grid_cells[i][j].configure(text=str(
                        new_number), bg=c.BACKGROUND_COLOR_DICT[str(new_number)],
                        fg=c.CELL_COLOR_DICT[str(new_number)])
        self.update_idletasks()

    def make_move(self):
        c = cfg.GAME_CONFIG
        state = self.model.power_grid(self.gamegrid)
        control_scores = self.model(np.expand_dims(state, axis=0))
        control_buttons = np.flip(np.argsort(control_scores),axis=1)
        for move in control_buttons[0]:
            prev_state = self.gamegrid.copy()
            temp_grid, changed, score = self.commands[move](prev_state)
            if not changed: #illegal move
                continue
            else:
                break
        self.gamegrid = prev_state.copy()
        self.score += score
        changed = True
        if game_over(self.gamegrid):
            changed = False
            self.grid_cells[1][1].configure(
                        text="Game", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
            self.grid_cells[1][2].configure(
                text="Over!", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
            self.grid_cells[2][1].configure(
                text="Score:", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
            self.grid_cells[2][2].configure(
                text=str(self.score), bg=c.BACKGROUND_COLOR_CELL_EMPTY)
        else:
            self.gamegrid = insert_new(self.gamegrid)
            self.update_grid_cells()

        if changed:
            self.after(7, self.make_move)


    def key_down(self, event):
        c = cfg.GAME_CONFIG
        key = repr(event.char)
        if key == c.KEY.BACK and len(self.history_matrices) > 1:
            self.gamegrid = self.history_matrices.pop()
            self.update_grid_cells()
            print('back on step total step:', len(self.history_matrices))
        elif key in self.commands:
            self.gamegrid, done, score = self.commands[repr(event.char)](self.gamegrid)
            self.score += score
            if done:
                self.gamegrid = insert_new(self.gamegrid)
                # record last move
                self.history_matrices.append(self.gamegrid.copy())
                self.update_grid_cells()
                done = False
                # if self.logic.game_over == 'win':
                #     self.grid_cells[1][1].configure(
                #         text="You", bg=BACKGROUND_COLOR_CELL_EMPTY)
                #     self.grid_cells[1][2].configure(
                #         text="Win!", bg=BACKGROUND_COLOR_CELL_EMPTY)
                if game_over(self.gamegrid):
                    self.grid_cells[1][1].configure(
                        text="Game", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                    self.grid_cells[1][2].configure(
                        text="Over!", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                    self.grid_cells[2][1].configure(
                        text="Score:", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                    self.grid_cells[2][2].configure(
                        text=str(self.score), bg=c.BACKGROUND_COLOR_CELL_EMPTY)


if __name__=='__main__':
    cfg_from_yaml_file('cfgs/SimpleRL.yaml', cfg)
    grid = Frame2048()