import argparse
import os

import torch

from agent import Agent
from game import SnakeGame

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def evaluate(model_path, headless=True, max_games=None):
    os.chdir(PROJECT_DIR)

    agent = Agent()
    agent.model.load_state_dict(torch.load(model_path, weights_only=True))
    agent.trainer.sync_target()

    game = SnakeGame(headless=headless, step_limit_per_length=None)

    print(f'Evaluating {model_path}')
    print(f'{"Game":>6}  {"Score":>6}')
    print('-' * 15)

    session_best = 0
    session_best_length = 3
    scores = []
    game_num = 1

    while True:
        if max_games is not None and game_num > max_games:
            break

        game.set_game_info(game_num, session_best_length)
        while True:
            state = game.get_state()
            action = agent.get_action(state, eval_mode=True)
            reward, done, score = game.play_step(action)
            if done:
                scores.append(score)
                session_best = max(session_best, score)
                session_best_length = max(session_best_length, len(game.snake))
                print(f'{game_num:>6}  {score:>6}')
                game.reset()
                game_num += 1
                break

        if headless and max_games is not None and game_num > max_games:
            break

    if scores:
        mean_score = sum(scores) / len(scores)
        print('-' * 15)
        print(f'Games: {len(scores)}  Mean: {mean_score:.2f}  Best: {max(scores)}')
    else:
        print('No games completed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate trained Snake RL model')
    parser.add_argument('--model', default='./model/model.pth')
    parser.add_argument('--visual', action='store_true',
                        help='Show pygame window (runs until closed if --games omitted)')
    parser.add_argument('--games', type=int, default=None,
                        help='Number of games to evaluate (default: infinite in visual mode)')
    args = parser.parse_args()
    evaluate(args.model, headless=not args.visual, max_games=args.games)