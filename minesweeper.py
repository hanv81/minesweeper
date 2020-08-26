from random import randint
import pygame
from agent import Agent

MINE = 9
SIZE = 25
PATH = './minesweeper/minesweeper-game/image/'
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
    def __init__(self, x, y, w, h, val):
        self.rect = pygame.rect.Rect(x, y, w, h)
        self.val = val
        self.x = x
        self.y = y
        self.visible = False
        self.flag = False

def restart(rows, cols, bombs):
    game(rows, cols, bombs)

def open_square(lst, square):
    square.visible = True
    i, j = square.x // SIZE, square.y // SIZE
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

def game(rows, cols, bombs):
    agent = Agent(rows, cols)
    table = create_table(rows, cols, bombs)

    w = cols * SIZE
    h = rows * SIZE
    screen = pygame.display.set_mode((w,h))

    lst = [[] for i in range(rows)]
    for i in range(rows):
        for j in range(cols):
            lst[i] += [Square(i * SIZE, j * SIZE, SIZE, SIZE, table[i][j])]
            screen.blit(GREY, (i * SIZE, j * SIZE))

    run = True
    win = False
    boom_cell = None
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    restart(rows, cols, bombs)
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
                    restart(rows, cols, bombs)
                elif event.key == pygame.K_ESCAPE:
                    run = False
                    pygame.quit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                run = False
                if event.button == 1:   # LEFT CLICK
                    restart(rows, cols, bombs)
                else:
                    pygame.quit()

def main():
    game(rows=10, cols=10, bombs=8)

if __name__ == "__main__":
    pygame.init()
    main()