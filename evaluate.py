import argparse
import os

import torch

from agent import Agent
from game import SnakeGame

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def evaluate(model_path, headless=True):
    os.chdir(PROJECT_DIR)

    agent = Agent()
    agent.model.load_state_dict(torch.load(model_path, weights_only=True))
    agent.trainer.sync_target()

    game = SnakeGame(headless=headless)

    print(f'Evaluating {model_path} — close pygame window to stop.')
    print(f'{"Game":>6}  {"Score":>6}')
    print('-' * 15)

    session_best_length = 3
    game_num = 1

    while True:
        game.set_game_info(game_num, session_best_length)
        while True:
            state = game.get_state()
            action = agent.get_action(state, eval_mode=True)
            reward, done, score = game.play_step(action)
            if done:
                session_best_length = max(session_best_length, len(game.snake))
                print(f'{game_num:>6}  {score:>6}')
                game.reset()
                game_num += 1
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate trained Snake RL model')
    parser.add_argument('--model', default='./model/model.pth')
    parser.add_argument('--visual', action='store_true')
    args = parser.parse_args()
    evaluate(args.model, headless=not args.visual)