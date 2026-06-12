import math
import pygame
import random
import numpy as np
from enum import Enum
from collections import namedtuple

pygame.init()

class Direction(Enum):
    RIGHT = 1
    LEFT  = 2
    UP    = 3
    DOWN  = 4

Point = namedtuple('Point', 'x, y')

BLOCK  = 20
SPEED  = 55          # normal play / evaluate
TRAIN_SPEED = 0      # 0 = uncapped FPS for fast visual training
WHITE  = (245, 245, 245)

BG        = (12, 18, 32)
GRID      = (22, 32, 48)
HEAD_COL  = (72, 220, 160)
TAIL_COL  = (24, 110, 88)
FOOD_COL  = (255, 92, 92)
FOOD_GLOW = (255, 150, 80)
ACCENT    = (100, 180, 255)

class SnakeGame:
    def __init__(self, w=640, h=480, headless=False, step_limit_per_length=100, fps=SPEED):
        self.w = w
        self.h = h
        self.headless = headless
        self.step_limit_per_length = step_limit_per_length
        self.fps = fps
        self.grid_w = w // BLOCK
        self.grid_h = h // BLOCK

        self.current_game = 1
        self.session_best_length = 3
        self.training_total = None
        self.training_best_score = 0
        self._anim_frame = 0
        self._eat_flash = 0

        if headless:
            self.display = None
            self.clock = None
        else:
            self.display = pygame.display.set_mode((w, h))
            pygame.display.set_caption('Snake RL')
            self.clock = pygame.time.Clock()
            self._title_font = pygame.font.SysFont('segoeui', 20, bold=True)
            self._hud_font = pygame.font.SysFont('segoeui', 16)

        self.reset()

    def set_game_info(self, current_game, session_best_length=None,
                      training_total=None, training_best_score=None):
        self.current_game = current_game
        if session_best_length is not None:
            self.session_best_length = session_best_length
        if training_total is not None:
            self.training_total = training_total
        if training_best_score is not None:
            self.training_best_score = training_best_score

    def reset(self):
        self.direction = Direction.RIGHT
        cx = self.w // 2
        cy = self.h // 2
        self.head = Point(cx, cy)
        self.snake = [
            Point(cx, cy),
            Point(cx - BLOCK, cy),
            Point(cx - 2 * BLOCK, cy),
        ]
        self.score = 0
        self.food = None
        self.frame_iteration = 0
        self._place_food()

    def _place_food(self):
        cells = [
            Point(x * BLOCK, y * BLOCK)
            for x in range(self.grid_w)
            for y in range(self.grid_h)
            if Point(x * BLOCK, y * BLOCK) not in self.snake
        ]
        if not cells:
            return
        self.food = random.choice(cells)

    def play_step(self, action):
        self.frame_iteration += 1

        if not self.headless:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

        self._move(action)
        self.snake.insert(0, self.head)

        reward = 0
        game_over = False

        timed_out = (
            self.step_limit_per_length is not None
            and self.frame_iteration > self.step_limit_per_length * len(self.snake)
        )
        if self.is_collision() or timed_out:
            game_over = True
            reward = -10
            if not self.headless:
                self._update_ui(game_over=True)
                self._tick()
            return reward, game_over, self.score

        if self.head == self.food:
            self.score += 1
            reward = 10
            self._eat_flash = 8
            self._place_food()
        else:
            self.snake.pop()

        if not self.headless:
            self._update_ui()
            self._tick()
        return reward, game_over, self.score

    def _tick(self):
        if self.fps > 0:
            self.clock.tick(self.fps)

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if pt.x >= self.w or pt.x < 0 or pt.y >= self.h or pt.y < 0:
            return True
        if pt in self.snake[1:]:
            return True
        return False

    def _move(self, action):
        clock_wise = [Direction.RIGHT, Direction.DOWN,
                      Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            new_dir = clock_wise[(idx + 1) % 4]
        else:
            new_dir = clock_wise[(idx - 1) % 4]

        self.direction = new_dir
        x, y = self.head.x, self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK
        elif self.direction == Direction.LEFT:
            x -= BLOCK
        elif self.direction == Direction.DOWN:
            y += BLOCK
        elif self.direction == Direction.UP:
            y -= BLOCK
        self.head = Point(x, y)

    def get_state(self):
        head = self.head
        pt_l = Point(head.x - BLOCK, head.y)
        pt_r = Point(head.x + BLOCK, head.y)
        pt_u = Point(head.x, head.y - BLOCK)
        pt_d = Point(head.x, head.y + BLOCK)

        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN

        state = [
            (dir_r and self.is_collision(pt_r)) or
            (dir_l and self.is_collision(pt_l)) or
            (dir_u and self.is_collision(pt_u)) or
            (dir_d and self.is_collision(pt_d)),

            (dir_u and self.is_collision(pt_r)) or
            (dir_d and self.is_collision(pt_l)) or
            (dir_l and self.is_collision(pt_u)) or
            (dir_r and self.is_collision(pt_d)),

            (dir_d and self.is_collision(pt_r)) or
            (dir_u and self.is_collision(pt_l)) or
            (dir_r and self.is_collision(pt_u)) or
            (dir_l and self.is_collision(pt_d)),

            dir_l, dir_r, dir_u, dir_d,

            self.food.x < head.x,
            self.food.x > head.x,
            self.food.y < head.y,
            self.food.y > head.y,
        ]
        return np.array(state, dtype=int)

    def _lerp_color(self, a, b, t):
        return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))

    def _draw_grid(self):
        for x in range(0, self.w, BLOCK):
            pygame.draw.line(self.display, GRID, (x, 0), (x, self.h), 1)
        for y in range(0, self.h, BLOCK):
            pygame.draw.line(self.display, GRID, (0, y), (self.w, y), 1)

    def _segment_rect(self, pt, inset=1):
        return pygame.Rect(pt.x + inset, pt.y + inset, BLOCK - 2 * inset, BLOCK - 2 * inset)

    def _draw_snake(self):
        length = len(self.snake)
        for i, pt in enumerate(self.snake):
            t = i / max(length - 1, 1)
            color = self._lerp_color(HEAD_COL, TAIL_COL, t)
            rect = self._segment_rect(pt, inset=1 if i else 0)
            pygame.draw.rect(self.display, color, rect, border_radius=4 if i == 0 else 3)
            if i == 0:
                self._draw_head_marker(rect)

    def _draw_head_marker(self, rect):
        cx = rect.centerx
        cy = rect.centery
        marker = (18, 40, 32)
        if self.direction == Direction.RIGHT:
            points = [(rect.right - 4, cy), (rect.right - 12, cy - 5), (rect.right - 12, cy + 5)]
        elif self.direction == Direction.LEFT:
            points = [(rect.left + 4, cy), (rect.left + 12, cy - 5), (rect.left + 12, cy + 5)]
        elif self.direction == Direction.UP:
            points = [(cx, rect.top + 4), (cx - 5, rect.top + 12), (cx + 5, rect.top + 12)]
        else:
            points = [(cx, rect.bottom - 4), (cx - 5, rect.bottom - 12), (cx + 5, rect.bottom - 12)]
        pygame.draw.polygon(self.display, marker, points)

    def _draw_food(self):
        if self.food is None:
            return
        pulse = 0.9 + 0.1 * math.sin(self._anim_frame * 0.2)
        inset = int((1 - pulse) * 3)
        rect = self._segment_rect(self.food, inset=inset)
        pygame.draw.rect(self.display, FOOD_GLOW, rect.inflate(4, 4), border_radius=6)
        pygame.draw.rect(self.display, FOOD_COL, rect, border_radius=5)

    def _draw_hud(self, game_over=False):
        panel = pygame.Surface((self.w, 52), pygame.SRCALPHA)
        panel.fill((18, 26, 42, 210))
        self.display.blit(panel, (0, 0))
        pygame.draw.line(self.display, ACCENT, (0, 52), (self.w, 52), 2)

        best_display = max(self.session_best_length, len(self.snake))

        if self.training_total:
            title = f'Training {self.current_game}/{self.training_total}'
        else:
            title = f'Game {self.current_game}'

        lines = [
            (self._title_font, title, ACCENT, (14, 10)),
            (self._hud_font, f'Score: {self.score}', WHITE, (14, 30)),
            (self._hud_font, f'Length: {len(self.snake)}', (200, 210, 220), (120, 30)),
            (self._hud_font, f'Best: {max(best_display, self.training_best_score)}',
             (180, 220, 180), (260, 30)),
        ]

        for font, text, color, pos in lines:
            surf = font.render(text, True, color)
            self.display.blit(surf, pos)

        if game_over:
            overlay = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
            overlay.fill((180, 30, 30, 60))
            self.display.blit(overlay, (0, 0))
            msg = self._title_font.render('Game Over', True, (255, 120, 120))
            rect = msg.get_rect(center=(self.w // 2, self.h // 2))
            self.display.blit(msg, rect)

    def _update_ui(self, game_over=False):
        self._anim_frame += 1
        if self._eat_flash > 0:
            self._eat_flash -= 1

        self.display.fill(BG)
        self._draw_grid()
        self._draw_food()
        self._draw_snake()
        self._draw_hud(game_over=game_over)
        pygame.display.flip()