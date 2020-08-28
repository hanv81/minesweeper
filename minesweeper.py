from random import randint
import pygame
from agent import Agent
import numpy as np
import time

MINE = 9
SIZE = 25
PATH = './minesweeper/image/'
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
HEURISTIC = True

def create_table(rows, cols, bombs):
    table = [[0] * cols for i in range(rows)]
    for i in range(bombs):
        while True:
            x = randint(0, len(table)-1)
            y = randint(0, len(table[0])-1)
            if table[x][y] != MINE:
                table[x][y] = MINE
                break
    for i in range(rows):
        for j in range(cols):
            if table[i][j] == MINE:
                table = check_down_left(table, i, j)
                table = check_down_right(table, i, j)
                table = check_down(table, i, j)
                table = check_up_left(table, i, j)
                table = check_up_right(table, i, j)
                table = check_up(table, i, j)
                table = check_left(table, i, j)
                table = check_right(table, i, j)
    return table

def check_down_left(table, x, y):
    if x+1 < len(table) and y-1 >= 0:
        if table[x+1][y-1] != MINE:
            table[x+1][y-1] += 1
    return table

def check_down_right(table, x, y):
    if x+1 < len(table) and y+1 < len(table[0]):
        if table[x+1][y+1] != MINE:
            table[x+1][y+1] += 1
    return table

def check_up_left(table, x, y):
    if x-1 >= 0 and y-1 >= 0:
        if table[x-1][y-1] != MINE:
            table[x-1][y-1] += 1
    return table

def check_up_right(table, x, y):
    if x-1 >= 0 and y+1 < len(table[0]):
        if table[x-1][y+1] != MINE:
            table[x-1][y+1] += 1
    return table

def check_up(table, x, y):
    if x-1 >= 0:
        if table[x-1][y] != MINE:
            table[x-1][y] += 1
    return table

def check_down(table, x, y):
    if x+1 < len(table):
        if table[x+1][y] != MINE:
            table[x+1][y] += 1
    return table

def check_left(table, x, y):
    if y-1 >= 0:
        if table[x][y-1] != MINE:
            table[x][y-1] += 1
    return table

def check_right(table, x, y):
    if y+1 < len(table[0]):
        if table[x][y+1] != MINE:
            table[x][y+1] += 1
    return table

class Square:
    def __init__(self, i, j, w, h, val):
        self.i = i
        self.j = j
        self.x = j*w
        self.y = i*h
        self.val = val
        self.visible = False
        self.flag = False
        self.rect = pygame.rect.Rect(self.x, self.y, w, h)

def restart(rows, cols, bombs, agent):
    game(rows, cols, bombs, agent)

def open_square(lst, square):
    square.visible = True
    i, j = square.i, square.j
    if i+1 < len(lst):
        if lst[i+1][j].visible == False and lst[i+1][j].flag == False:
            lst[i+1][j].visible = True
            if lst[i+1][j].val == 0:
                open_square(lst, lst[i+1][j])
        if j+1 < len(lst[0]):
            if lst[i+1][j+1].visible == False and lst[i+1][j+1].flag == False:
                lst[i+1][j+1].visible = True
                if lst[i+1][j+1].val == 0:
                    open_square(lst, lst[i+1][j+1])
        if j-1 >= 0:
            if lst[i+1][j-1].visible == False and lst[i+1][j-1].flag == False:
                lst[i+1][j-1].visible = True
                if lst[i+1][j-1].val == 0:
                    open_square(lst, lst[i+1][j-1])
    if i-1 >= 0:
        if lst[i-1][j].visible == False and lst[i-1][j].flag == False:
            lst[i-1][j].visible = True
            if lst[i-1][j].val == 0:
                open_square(lst, lst[i-1][j])
        if j+1 < len(lst[0]):
            if lst[i-1][j+1].visible == False and lst[i-1][j+1].flag == False:
                lst[i-1][j+1].visible = True
                if lst[i-1][j+1].val == 0:
                    open_square(lst, lst[i-1][j+1])
        if j-1 >= 0:
            if lst[i-1][j-1].visible == False and lst[i-1][j-1].flag == False:
                lst[i-1][j-1].visible = True
                if lst[i-1][j-1].val == 0:
                    open_square(lst, lst[i-1][j-1])
    if j-1 >= 0:
        if lst[i][j-1].visible == False and lst[i][j-1].flag == False:
            lst[i][j-1].visible = True
            if lst[i][j-1].val == 0:
                open_square(lst, lst[i][j-1])
    if j+1 < len(lst[0]):
        if lst[i][j+1].visible == False and lst[i][j+1].flag == False:
            lst[i][j+1].visible = True
            if lst[i][j+1].val == 0:
                open_square(lst, lst[i][j+1])

def game(rows, cols, bombs, agent):
    table = create_table(rows, cols, bombs)

    w = cols * SIZE
    h = rows * SIZE
    screen = pygame.display.set_mode((w,h))

    lst = [[] for i in range(rows)]
    for i in range(rows):
        for j in range(cols):
            square = Square(i, j, SIZE, SIZE, table[i][j])
            lst[i] += [square]
            screen.blit(GREY, (square.x, square.y))

    run = True
    win = False
    boom_cell = None
    auto = False
    point = 0
    while run:
        if auto:    # agent play
            state = []
            clicked = []
            if point == 0:
                action = randint(0, rows * cols - 1)
                i = action // len(lst[0])
                j = action % len(lst[0])
                square = lst[i][j]
            else:
                if HEURISTIC:
                    lst = heuristic(lst)

                action = 0
                for i in lst:
                    row = []
                    for j in i:
                        if j.visible or j.flag:
                            row.append(j.val)
                            clicked.append(action)
                        else:
                            row.append(-1)
                        action += 1
                    state.append(row)
                qs = agent.predict(np.array(state))[0]
                for cell in clicked:
                    qs[cell] = np.min(qs)
                if np.max(qs) > np.min(qs):
                    action = np.argmax(qs)
                    i = action // len(lst[0])
                    j = action % len(lst[0])
                    square = lst[i][j]
                else:
                    print('no max q, just random')
                    cells_to_click = []
                    for i in range(rows):
                        for j in range(cols):
                            if not lst[j][j].visible and not lst[i][j].flag:
                                cells_to_click.append(lst[i][j])
                    square = np.random.sample(cells_to_click, 1)[0]

            if not square.flag:
                if square.val == MINE:
                    print('BOMBS')
                    boom_cell = square
                    run = False
                else:
                    point += 1
                square.visible = True
                if square.val == 0:
                    open_square(lst, square)
            time.sleep(1)

        else:   # user play
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    pygame.quit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        restart(rows, cols, bombs, agent)
                    elif event.key == pygame.K_a:
                        auto = True
                    elif event.key == pygame.K_ESCAPE:
                        run = False
                        pygame.quit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    r = pygame.rect.Rect(pygame.mouse.get_pos(), (1,1))
                    for i in lst:
                        for j in i:
                            if j.rect.colliderect(r):
                                if event.button == 1:   # LEFT CLICK
                                    if not j.flag:
                                        if j.val == MINE:
                                            print('BOMBS')
                                            boom_cell = j
                                            run = False
                                        else:
                                            point += 1
                                        j.visible = True
                                        if j.val == 0:
                                            open_square(lst, j)

                                elif event.button == 3: # RIGHT CLICK
                                    if not j.visible:
                                        j.flag = not j.flag

        for i in lst:
            for j in i:
                if j.visible:
                    screen.blit(NUMBERS[j.val], (j.x, j.y))
                if j.flag:
                    screen.blit(FLAG, (j.x, j.y))
                if not j.flag and not j.visible:
                    screen.blit(GREY, (j.x, j.y))
        cnt = 0
        for i in lst:
            for j in i:
                if j.visible and j.val != MINE:
                    cnt += 1
        if cnt == rows * cols - bombs:
            run = False
            win = True
            print('WIN')
        pygame.display.update()

    print('point:', point)
    if win:
        for i in lst:
            for j in i:
                if not j.visible:
                    screen.blit(FLAG, (j.x, j.y))
        width, height = GLASSES.get_rect().size
        screen.blit(GLASSES, ((w-width)//2, (h-height)//2))
    else:
        for i in lst:
            for j in i:
                if j.val == MINE:
                    screen.blit(BOMB, (j.x, j.y))
        width, height = SAD.get_rect().size
        screen.blit(BOOM, (boom_cell.x, boom_cell.y))
        screen.blit(SAD, ((w-width)//2, (h-height)//2))
    pygame.display.update()

    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    run = False
                    restart(rows, cols, bombs, agent)
                elif event.key == pygame.K_ESCAPE:
                    run = False
                    pygame.quit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                run = False
                if event.button == 1:   # LEFT CLICK
                    restart(rows, cols, bombs, agent)
                else:
                    pygame.quit()

def heuristic(lst):
    rows = len(lst)
    cols = len(lst[0])
    for i in range(rows):
        for j in range(cols):
            if not lst[i][j].visible:
                lst[i][j].flag = False
    for i in range(rows):
        for j in range(cols):
            square = lst[i][j]
            if square.visible and square.val > 0:  # square is open
                neibors = []
                neibors_flag = []
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

def main():
    rows=10
    cols=10
    bombs=10
    agent = Agent()
    agent.load_model('./minesweeper/model/model.h5')
    game(rows, cols, bombs, agent)

if __name__ == "__main__":
    pygame.init()
    main()