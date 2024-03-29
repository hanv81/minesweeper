import pygame
import numpy as np
import time
import random

MINE = 9
PATH = './game/image/'
GREY = pygame.image.load(PATH + 'grey.png')
FLAG = pygame.image.load(PATH + 'flag.png')
BOOM = pygame.image.load(PATH + 'boom.png')
GLASSES = pygame.image.load(PATH + 'sun-glasses.png')
SAD = pygame.image.load(PATH + 'sun-sad.png')
NUMBERS = [pygame.image.load(PATH + str(i) + '.png') for i in range(10)]
SIZE = GREY.get_width()

class Square:
    def __init__(self, i, r, c, w, h, val):
        self.i, self.x, self.y, self.val = i, c*w, r*h, val
        self.visible, self.flag = False, False
        self.rect = pygame.rect.Rect(self.x, self.y, w, h)

class Game:
    def __init__(self, rows, cols, mines, agent):
        self.rows, self.cols, self.mines, self.agent = rows, cols, mines, agent
        if agent is not None:
            self.agent.action_size = rows * cols
        self.screen = pygame.display.set_mode((self.cols * SIZE, self.rows * SIZE))
        self.init_game()

    def init_game(self):
        self.run, self.win, self.auto = True, False, False
        self.point, self.open = 0, 0
        self.boom_cell = None

        mines = random.sample(range(self.rows*self.cols), self.mines)
        self.squares = []
        for i in range(self.rows*self.cols):
            val = MINE if i in mines else 0
            square = Square(i, i // self.cols, i % self.cols, SIZE, SIZE, val)
            self.squares += [square]

        for square in self.squares:
            if square.val != MINE:
                r,c = square.i // self.cols, square.i % self.cols
                rc = (r, c+1), (r, c-1), (r+1, c), (r+1, c+1), (r+1, c-1), (r-1, c), (r-1, c+1), (r-1, c-1)
                for r,c in rc:
                    if 0 <= r < self.rows and 0 <= c < self.cols and self.squares[r * self.cols + c].val == MINE:
                        square.val += 1

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
                    self.screen.blit(NUMBERS[MINE], (square.x, square.y))

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
            r,c = square.i // self.cols, square.i % self.cols
            rc = [(r, c+1), (r, c-1), (r+1, c), (r+1, c+1), (r+1, c-1), (r-1, c), (r-1, c+1), (r-1, c-1)]
            for r,c in rc:
                if 0 <= r < self.rows and 0 <= c < self.cols:
                    square = self.squares[r * self.cols + c]
                    if not square.visible and not square.flag:
                        self.open_square_recursive(square)

    def open_square(self, square):
        if square.val == MINE:
            print('BOMBS', self.point)
            self.boom_cell = square
            self.run, self.auto = False, False
        else:
            self.point += 1
            self.open_square_recursive(square)
            if self.open == self.rows * self.cols - self.mines:
                self.run, self.win, self.auto = False, True, False
                print('WIN', self.point)

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
                elif event.key == pygame.K_f:
                    self.flagging()
                elif event.key == pygame.K_ESCAPE:
                    self.run = False
                    pygame.quit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if self.run:
                    r = pygame.rect.Rect(pygame.mouse.get_pos(), (1,1))
                    for square in self.squares:
                        if square.rect.colliderect(r) and not square.visible:
                            if event.button == pygame.BUTTON_LEFT:
                                if not square.flag:
                                    self.open_square(square)
                            elif event.button == pygame.BUTTON_RIGHT:
                                square.flag = not square.flag
                            break
                else:
                    self.init_game()

    def flagging(self):
        for square in self.squares:
            if not square.visible:
                square.flag = False

        for square in self.squares:
            if square.visible and square.val > 0:  # square is open and have mines around
                neibors, neibors_flag = [], []
                r,c = square.i // self.cols, square.i % self.cols
                rc = (r, c+1), (r, c-1), (r+1, c), (r+1, c+1), (r+1, c-1), (r-1, c), (r-1, c+1), (r-1, c-1)
                for r,c in rc:
                    if 0 <= r < self.rows and 0 <= c < self.cols:
                        sqr = self.squares[r*self.cols + c]
                        if not sqr.visible:
                            if sqr.flag:
                                neibors_flag.append(sqr)
                            else:
                                neibors.append(sqr)
                if len(neibors) == square.val - len(neibors_flag):
                    for sqr in neibors:
                        sqr.flag = True
        time.sleep(0.5)

    def agent_action(self):
        state = np.full((self.rows, self.cols), -1)
        clicked = []
        for square in self.squares:
            if square.visible:
                state[square.i // self.cols, square.i % self.cols] = square.val
                clicked.append(square.i)

        if self.agent.is_value_based():
            qs = self.agent.predict(state)[0]
            for square in self.squares:
                if square.i in clicked or square.flag:
                    qs[square.i] = np.min(qs)
            if np.max(qs) > np.min(qs):
                return self.squares[np.argmax(qs)]

            print('no max q, just random')
            cells_to_click = []
            for square in self.squares:
                if not square.visible and not square.flag:
                    cells_to_click.append(square)
            return random.sample(cells_to_click, 1)[0]
        else:
            while True:
                i = self.agent.act(state)
                if not self.squares[i].visible:
                    return self.squares[i]

    def agent_play(self):
        if self.point == 0:
            i = random.randint(0, self.rows * self.cols - 1)
            square = self.squares[i]
        else:
            if self.agent.is_value_based():
                self.flagging()
            square = self.agent_action()

        if not square.visible and not square.flag:
            self.open_square(square)
        time.sleep(1)

    def play_game(self):
        pygame.init()
        self.start()

if __name__ == "__main__":
    game = Game(5, 5, 3, None)
    game.play_game()