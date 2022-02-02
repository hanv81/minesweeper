from random import randint
import pygame
import numpy as np
import time
import random

MINE = 9
HEURISTIC = True
PATH = './game/image/'
GREY = pygame.image.load(PATH + 'grey.png')
ZERO = pygame.image.load(PATH + '0.png')
ONE = pygame.image.load(PATH + '1.png')
TWO = pygame.image.load(PATH + '2.png')
THREE = pygame.image.load(PATH + '3.png')
FOUR = pygame.image.load(PATH + '4.png')
FIVE = pygame.image.load(PATH + '5.png')
SIX = pygame.image.load(PATH + '6.png')
SEVEN = pygame.image.load(PATH + '7.png')
EIGHT = pygame.image.load(PATH + '8.png')
BOMB = pygame.image.load(PATH + 'bomb.png')
FLAG = pygame.image.load(PATH + 'flag.png')
BOOM = pygame.image.load(PATH + 'boom.png')
GLASSES = pygame.image.load(PATH + 'sun-glasses.png')
SAD = pygame.image.load(PATH + 'sun-sad.png')
NUMBERS = [ZERO, ONE, TWO, THREE, FOUR, FIVE, SIX, SEVEN, EIGHT, BOMB]
SIZE = GREY.get_width()

class Square:
    def __init__(self, i, j, w, h, val):
        self.i, self.j = i, j
        self.x, self.y = j*w, i*h
        self.val = val
        self.visible, self.flag = False, False
        self.rect = pygame.rect.Rect(self.x, self.y, w, h)

class Game:
    def __init__(self, rows, cols, mines, agent):
        self.rows = rows
        self.cols = cols
        self.mines = mines
        self.agent = agent
        self.cells = rows * cols
        self.screen = pygame.display.set_mode((self.cols * SIZE, self.rows * SIZE))
        self.init_game()

    def init_game(self):
        self.run, self.win, self.auto = True, False, False
        self.point, self.open = 0, 0
        self.boom_cell = None

        mines = random.sample(range(self.cells), self.mines)
        self.squares = []
        for i in range(self.cells):
            val = MINE if i in mines else 0
            square = Square(i // self.cols, i % self.cols, SIZE, SIZE, val)
            self.squares += [square]
        for i in range(self.cells):
            if self.squares[i].val != MINE:
                r,c = i // self.cols, i % self.cols
                rc = [(r, c+1), (r, c-1), (r+1, c), (r+1, c+1), (r+1, c-1), (r-1, c), (r-1, c+1), (r-1, c-1)]
                for r,c in rc:
                    if 0 <= r <self.rows and 0 <= c < self.cols and self.squares[r * self.cols + c].val == MINE:
                        self.squares[i].val += 1

    def start(self):
        while True:
            if self.auto:    # agent play
                if self.run:
                    self.agent_play()
            else:   # user play
                self.user_play()
            self.paint()

    def paint(self):
        for square in self.squares:
            if square.visible:
                self.screen.blit(NUMBERS[square.val], (square.x, square.y))
            else:
                if square.flag or self.win:
                    self.screen.blit(FLAG, (square.x, square.y))
                elif self.run or square.val != MINE:
                    self.screen.blit(GREY, (square.x, square.y))
                else:
                    self.screen.blit(BOMB, (square.x, square.y))

        if not self.run:
            if self.win:
                width, height = GLASSES.get_rect().size
                self.screen.blit(GLASSES, ((self.screen.get_width()-width)//2, (self.screen.get_height()-height)//2))
            else:
                width, height = SAD.get_rect().size
                self.screen.blit(BOOM, (self.boom_cell.x, self.boom_cell.y))
                self.screen.blit(SAD, ((self.screen.get_width()-width)//2, (self.screen.get_height()-height)//2))

        pygame.display.update()

    def open_square_recursive(self, square):
        self.open += 1
        square.visible = True
        if square.val == 0:
            r,c = square.i, square.j
            rc = [(r, c+1), (r, c-1), (r+1, c), (r+1, c+1), (r+1, c-1), (r-1, c), (r-1, c+1), (r-1, c-1)]
            for r,c in rc:
                if 0 <=r <self.rows and 0 <=c< self.cols:
                    square = self.squares[r * self.cols + c]
                    if not square.visible and not square.flag:
                        self.open_square_recursive(square)

    def open_square(self, square):
        if square.val == MINE:
            print('BOMBS')
            self.boom_cell = square
            self.run, self.auto = False, False
        else:
            self.point += 1
            self.open_square_recursive(square)
            if self.open == self.rows * self.cols - self.mines:
                self.run, self.win, self.auto = False, True, False
                print('WIN')

    def user_play(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.run = False
                pygame.quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    if not self.run:
                        self.init_game()
                elif event.key == pygame.K_a:
                    self.auto = True
                elif event.key == pygame.K_ESCAPE:
                    self.run = False
                    pygame.quit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if self.run:
                    r = pygame.rect.Rect(pygame.mouse.get_pos(), (1,1))
                    for square in self.squares:
                        if square.rect.colliderect(r) and not square.visible:
                            if event.button == 1:   # LEFT CLICK
                                if not square.flag:
                                    self.open_square(square)
                            elif event.button == 3: # RIGHT CLICK
                                square.flag = not square.flag
                            break
                else:
                    self.init_game()

    def heuristic(self):
        for i,j in self.ij:
            if not self.squares[i][j].visible:
                self.squares[i][j].flag = False
        for i,j in self.ij:
            square = self.squares[i][j]
            if square.visible and square.val > 0:  # square is open
                neibors, neibors_flag = [], []
                rc = [(i, j+1), (i, j-1), (i+1, j), (i+1, j+1), (i+1, j-1), (i-1, j), (i-1, j+1), (i-1, j-1)]
                for r,c in rc:
                    if 0<=r<self.rows and 0<=c<self.cols:
                        if not self.squares[r][c].visible:
                            if self.squares[r][c].flag:
                                neibors_flag.append(self.squares[r][c])
                            else:
                                neibors.append(self.squares[r][c])
                if len(neibors) == square.val - len(neibors_flag):
                    for n in neibors:
                        n.flag = True
        time.sleep(0.5)

    def agent_action(self):
        action = 0
        state = []
        clicked = []
        for row in self.squares:
            r = []
            for square in row:
                if square.visible or square.flag:
                    r.append(square.val)
                    clicked.append(action)
                else:
                    r.append(-1)    # UNKNOWN
                action += 1
            state.append(r)
        qs = self.agent.predict(np.array(state))[0]
        for cell in clicked:
            qs[cell] = np.min(qs)
        if np.max(qs) > np.min(qs):
            action = np.argmax(qs)
            i, j = action // self.cols, action % self.cols
            square = self.squares[i][j]
        else:
            print('no max q, just random')
            cells_to_click = []
            for i,j in self.ij:
                if not self.squares[j][j].visible and not self.squares[i][j].flag:
                    cells_to_click.append(self.squares[i][j])
            square = random.sample(cells_to_click, 1)[0]
        return square

    def agent_play(self):
        if self.point == 0:
            action = randint(0, self.rows * self.cols - 1)
            i, j = action // self.cols, action % self.cols
            square = self.squares[i][j]
        else:
            if HEURISTIC:
                self.heuristic()
            square = self.agent_action()

        if not square.flag:
            self.open_square(square)
        time.sleep(1)

    def play_game(self):
        pygame.init()
        self.start()

if __name__ == "__main__":
    game = Game(5, 5, 3, None)
    game.play_game()