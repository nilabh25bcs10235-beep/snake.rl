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
SPEED  = 55
WHITE  = (245, 245, 245)

BG        = (12, 18, 32)
GRID      = (22, 32, 48)
HEAD_COL  = (72, 220, 160)
TAIL_COL  = (24, 110, 88)
FOOD_COL  = (255, 92, 92)
FOOD_GLOW = (255, 150, 80)
ACCENT    = (100, 180, 255)

class SnakeGame:
    def __init__(self, w=640, h=480, headless=False):
        self.w = w
        self.h = h
        self.headless = headless
        self.grid_w = w // BLOCK
        self.grid_h = h // BLOCK

        self.current_game = 1
        self.session_best_length = 3
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

    def set_game_info(self, current_game, session_best_length=None):
        self.current_game = current_game
        if session_best_length is not None:
            self.session_best_length = session_best_length

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w // 2, self.h // 2)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK, self.head.y),
            Point(self.head.x - 2 * BLOCK, self.head.y),
        ]
        self.score = 0
        self.food = None
        self.frame_iteration = 0
        self._place_food()

    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK) // BLOCK) * BLOCK
        y = random.randint(0, (self.h - BLOCK) // BLOCK) * BLOCK
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

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

        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            if not self.headless:
                self._update_ui(game_over=True)
                self.clock.tick(SPEED)
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
            self.clock.tick(SPEED)
        return reward, game_over, self.score

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

    def _draw_snake(self):
        length = len(self.snake)
        for i, pt in enumerate(self.snake):
            cx = pt.x + BLOCK // 2
            cy = pt.y + BLOCK // 2
            t = i / max(length - 1, 1)
            color = self._lerp_color(HEAD_COL, TAIL_COL, t)
            radius = BLOCK // 2 - 1

            if i == 0:
                pygame.draw.circle(self.display, (20, 80, 60), (cx, cy), radius + 2)
                pygame.draw.circle(self.display, color, (cx, cy), radius)
                self._draw_eyes(cx, cy)
            else:
                pygame.draw.circle(self.display, color, (cx, cy), radius - 1)

    def _draw_eyes(self, cx, cy):
        eye_offset = 5
        eye_r = 3
        pupil_r = 1
        white = (240, 250, 245)
        pupil = (20, 30, 40)

        if self.direction == Direction.RIGHT:
            positions = [(cx + 4, cy - eye_offset), (cx + 4, cy + eye_offset)]
        elif self.direction == Direction.LEFT:
            positions = [(cx - 4, cy - eye_offset), (cx - 4, cy + eye_offset)]
        elif self.direction == Direction.UP:
            positions = [(cx - eye_offset, cy - 4), (cx + eye_offset, cy - 4)]
        else:
            positions = [(cx - eye_offset, cy + 4), (cx + eye_offset, cy + 4)]

        for ex, ey in positions:
            pygame.draw.circle(self.display, white, (ex, ey), eye_r)
            pygame.draw.circle(self.display, pupil, (ex, ey), pupil_r)

    def _draw_food(self):
        cx = self.food.x + BLOCK // 2
        cy = self.food.y + BLOCK // 2
        pulse = 0.88 + 0.12 * math.sin(self._anim_frame * 0.2)
        glow_r = int((BLOCK // 2 + 4) * pulse)
        core_r = int((BLOCK // 2 - 2) * pulse)

        glow_surf = pygame.Surface((glow_r * 2, glow_r * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*FOOD_GLOW, 70), (glow_r, glow_r), glow_r)
        self.display.blit(glow_surf, (cx - glow_r, cy - glow_r))

        pygame.draw.circle(self.display, FOOD_COL, (cx, cy), core_r)
        pygame.draw.circle(self.display, (255, 200, 160), (cx - 3, cy - 3), 3)

    def _draw_hud(self, game_over=False):
        panel = pygame.Surface((self.w, 52), pygame.SRCALPHA)
        panel.fill((18, 26, 42, 210))
        self.display.blit(panel, (0, 0))
        pygame.draw.line(self.display, ACCENT, (0, 52), (self.w, 52), 2)

        best_display = max(self.session_best_length, len(self.snake))

        lines = [
            (self._title_font, f'Game {self.current_game}', ACCENT, (14, 10)),
            (self._hud_font, f'Length: {len(self.snake)}', WHITE, (14, 30)),
            (self._hud_font, f'Best: {best_display}', (180, 220, 180), (200, 30)),
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