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

def create_table(rows, cols, mines):
    table = [[0] * cols for _ in range(rows)]
    while mines > 0:
        x = randint(0, len(table)-1)
        y = randint(0, len(table[0])-1)
        if table[x][y] != MINE:
            table[x][y] = MINE
            mines -= 1

    for i in range(rows):
        for j in range(cols):
            if table[i][j] != MINE:
                ij = [(i, j+1), (i, j-1), (i+1, j), (i+1, j+1), (i+1, j-1), (i-1, j), (i-1, j+1), (i-1, j-1)]
                for (ii, jj) in ij:
                    if 0 <= ii < len(table) and 0 <= jj < len(table[0]) and table[ii][jj] == MINE:
                        table[i][j] += 1
    return table

class Square:
    def __init__(self, i, j, w, h, val):
        self.i, self.j = i, j
        self.x, self.y = j*w, i*h
        self.val = val
        self.visible, self.flag = False, False
        self.rect = pygame.rect.Rect(self.x, self.y, w, h)

def open_square(lst, square):
    square.visible = True
    i, j = square.i, square.j
    if square.val == 0:
        ij = [(i, j+1), (i, j-1), (i+1, j), (i+1, j+1), (i+1, j-1), (i-1, j), (i-1, j+1), (i-1, j-1)]
        for (i,j) in ij:
            if 0 <= i < len(lst) and 0 <= j < len(lst[0]):
                if not lst[i][j].visible and not lst[i][j].flag:
                    open_square(lst, lst[i][j])

def init_squares(rows, cols, mines):
    table = create_table(rows, cols, mines)
    squares = [[] for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            square = Square(i, j, SIZE, SIZE, table[i][j])
            squares[i] += [square]
    return squares

def init_game(rows, cols, mines):
    w = cols * SIZE
    h = rows * SIZE
    screen = pygame.display.set_mode((w,h))
    squares = init_squares(rows, cols, mines)
    return squares, screen

def start(rows, cols, mines, agent):
    squares, screen = init_game(rows, cols, mines)

    run, win, auto = True, False, False
    boom_cell = None
    point = 0
    while True:
        if auto:    # agent play
            if point == 0:
                action = randint(0, rows * cols - 1)
                i, j = action // cols, action % cols
                square = squares[i][j]
            else:
                if HEURISTIC:
                    squares = heuristic(squares)
                square = agent_action(agent, squares, rows, cols)

            if not square.flag:
                if square.val == MINE:
                    print('BOMBS')
                    boom_cell = square
                    run = False
                else:
                    point += 1
                    open_square(squares, square)
            time.sleep(1)

        else:   # user play
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    pygame.quit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        if not run:
                            squares = init_squares(rows, cols, mines)
                            point, run, win = 0, True, False
                    elif event.key == pygame.K_a:
                        auto = True
                    elif event.key == pygame.K_ESCAPE:
                        run = False
                        pygame.quit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if run:
                        r = pygame.rect.Rect(pygame.mouse.get_pos(), (1,1))
                        for row in squares:
                            for square in row:
                                if square.rect.colliderect(r):
                                    if event.button == 1:   # LEFT CLICK
                                        if not square.flag:
                                            if square.val == MINE:
                                                print('BOMBS')
                                                boom_cell = square
                                                run = False
                                            else:
                                                point += 1
                                                open_square(squares, square)

                                    elif event.button == 3: # RIGHT CLICK
                                        if not square.visible:
                                            square.flag = not square.flag
                    else:
                        squares = init_squares(rows, cols, mines)
                        point, run, win = 0, True, False

        if run: # check if win
            cnt = 0
            for row in squares:
                for square in row:
                    if square.visible:
                        cnt += 1
            if cnt == rows * cols - mines:
                run, win = False, True
                print('WIN')

        paint(squares, screen, boom_cell, run, win)

def paint(squares, screen, boom_cell, run, win):
    for row in squares:
        for square in row:
            if square.visible:
                screen.blit(NUMBERS[square.val], (square.x, square.y))
            else:
                if square.flag or win:
                    screen.blit(FLAG, (square.x, square.y))
                elif run or square.val != MINE:
                    screen.blit(GREY, (square.x, square.y))
                else:
                    screen.blit(BOMB, (square.x, square.y))

    if not run:
        if win:
            width, height = GLASSES.get_rect().size
            screen.blit(GLASSES, ((screen.get_width()-width)//2, (screen.get_height()-height)//2))
        else:
            width, height = SAD.get_rect().size
            screen.blit(BOOM, (boom_cell.x, boom_cell.y))
            screen.blit(SAD, ((screen.get_width()-width)//2, (screen.get_height()-height)//2))

    pygame.display.update()

def listen_event(rows, cols, mines, agent):
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    start(rows, cols, mines, agent)
                    return
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:   # LEFT CLICK
                    start(rows, cols, mines, agent)
                else:
                    pygame.quit()
                return

def heuristic(lst):
    rows, cols = len(lst), len(lst[0])
    for i in range(rows):
        for j in range(cols):
            if not lst[i][j].visible:
                lst[i][j].flag = False
    for i in range(rows):
        for j in range(cols):
            square = lst[i][j]
            if square.visible and square.val > 0:  # square is open
                neibors, neibors_flag = [], []
                for r in range(i-1, i+2):
                    for c in range(j-1, j+2):
                        if 0<=r<rows and 0<=c<cols:
                            if not lst[r][c].visible:
                                if lst[r][c].flag:
                                    neibors_flag.append(lst[r][c])
                                else:
                                    neibors.append(lst[r][c])
                if len(neibors) == square.val - len(neibors_flag):
                    for n in neibors:
                        n.flag = True
    time.sleep(0.5)
    return lst

def agent_action(agent, squares, rows, cols):
    action = 0
    state = []
    clicked = []
    for row in squares:
        r = []
        for square in row:
            if square.visible or square.flag:
                r.append(square.val)
                clicked.append(action)
            else:
                r.append(-1)    # UNKNOWN
            action += 1
        state.append(r)
    qs = agent.predict(np.array(state))[0]
    for cell in clicked:
        qs[cell] = np.min(qs)
    if np.max(qs) > np.min(qs):
        action = np.argmax(qs)
        i, j = action // cols, action % cols
        square = squares[i][j]
    else:
        print('no max q, just random')
        cells_to_click = []
        for i in range(rows):
            for j in range(cols):
                if not squares[j][j].visible and not squares[i][j].flag:
                    cells_to_click.append(squares[i][j])
        square = random.sample(cells_to_click, 1)[0]
    return square

def play_game(agent, rows, cols, mines):
    pygame.init()
    start(rows, cols, mines, agent)

if __name__ == "__main__":
    play_game(None, 10, 10, 10)