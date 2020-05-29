from tkinter import Frame, Label, CENTER
from logic2048 import *

class Frame2048(Frame):
    def __init__(self, human_player=True):
        self.humanPlayer = human_player
        self.logic = Logic2048(gridsize=GRID_LEN)
        if human_player:
            Frame.__init__(self)
            self.grid()
            self.master.title('2048')
            self.master.bind("<Key>", self.key_down)

            self.commands = {KEY_UP: self.logic.up, KEY_DOWN: self.logic.down,
                        KEY_LEFT: self.logic.left, KEY_RIGHT: self.logic.right,
                        KEY_UP_ALT: self.logic.up, KEY_DOWN_ALT: self.logic.down,
                        KEY_LEFT_ALT: self.logic.left, KEY_RIGHT_ALT: self.logic.right}

            self.grid_cells = []    # GUI elements
            self.init_grid()
            self.init_matrix()
            self.update_grid_cells()

            self.mainloop()

    def init_grid(self):
        background = Frame(self, bg=BACKGROUND_COLOR_GAME,
                           width=SIZE, height=SIZE)
        background.grid()

        for i in range(GRID_LEN):
            grid_row = []
            for j in range(GRID_LEN):
                cell = Frame(background, bg=BACKGROUND_COLOR_CELL_EMPTY,
                             width=SIZE / GRID_LEN,
                             height=SIZE / GRID_LEN)
                cell.grid(row=i, column=j, padx=GRID_PADDING,
                          pady=GRID_PADDING)
                t = Label(master=cell, text="",
                          bg=BACKGROUND_COLOR_CELL_EMPTY,
                          justify=CENTER, font=FONT, width=5, height=2)
                t.grid()
                grid_row.append(t)

            self.grid_cells.append(grid_row)
        return
    
    def init_matrix(self):
        self.logic.insert_new()
        self.logic.insert_new()
        self.history_matrices = list()

    def update_grid_cells(self):
        for i in range(GRID_LEN):
            for j in range(GRID_LEN):
                new_number = self.logic.grid[i][j]
                if new_number == 0:
                    self.grid_cells[i][j].configure(
                        text="", bg=BACKGROUND_COLOR_CELL_EMPTY)
                else:
                    self.grid_cells[i][j].configure(text=str(
                        new_number), bg=BACKGROUND_COLOR_DICT[new_number],
                        fg=CELL_COLOR_DICT[new_number])
        self.update_idletasks()

    def key_down(self, event):
        key = repr(event.char)
        if key == KEY_BACK and len(self.history_matrices) > 1:
            self.logic.grid = self.history_matrices.pop()
            self.update_grid_cells()
            print('back on step total step:', len(self.history_matrices))
        elif key in self.commands:
            done, score = self.commands[repr(event.char)]()
            if done:
                self.logic.insert_new()
                # record last move
                self.history_matrices.append(self.logic.grid.copy())
                self.update_grid_cells()
                done = False
                # if self.logic.game_over == 'win':
                #     self.grid_cells[1][1].configure(
                #         text="You", bg=BACKGROUND_COLOR_CELL_EMPTY)
                #     self.grid_cells[1][2].configure(
                #         text="Win!", bg=BACKGROUND_COLOR_CELL_EMPTY)
                if self.logic.game_over():
                    self.grid_cells[1][1].configure(
                        text="You", bg=BACKGROUND_COLOR_CELL_EMPTY)
                    self.grid_cells[1][2].configure(
                        text="Lose!", bg=BACKGROUND_COLOR_CELL_EMPTY)

if __name__=='__main__':
    grid = Frame2048()