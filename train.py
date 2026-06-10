import argparse
import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from agent import Agent
from game import SnakeGame

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ROLLING_WINDOW = 100


def plot(scores, mean_scores, rolling_means):
    plt.clf()
    plt.title('Snake RL — Training Progress')
    plt.xlabel('Number of games')
    plt.ylabel('Score')
    plt.plot(scores, label='Score', alpha=0.25, linewidth=0.8)
    plt.plot(mean_scores, label='Mean score', linewidth=1.5)
    plt.plot(rolling_means, label=f'Rolling mean ({ROLLING_WINDOW})', linewidth=2.5)
    plt.legend()
    plt.ylim(ymin=0)
    if scores:
        plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
        plt.text(len(rolling_means) - 1, rolling_means[-1],
                 f'{rolling_means[-1]:.1f}')
    plt.savefig(os.path.join(PROJECT_DIR, 'progress.png'), dpi=100)


def rolling_mean(values, window):
    if not values:
        return []
    out = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        chunk = values[start:i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def format_elapsed(seconds):
    minutes, secs = divmod(int(seconds), 60)
    return f'{minutes}m {secs}s'


def train(headless=True, plot_every=25):
    os.chdir(PROJECT_DIR)
    start_time = time.time()

    scores = []
    mean_scores = []
    rolling_means = []
    total_score = 0
    best_score = 0
    best_length = 3
    best_rolling = 0

    agent = Agent()
    game = SnakeGame(headless=headless)

    mode = 'headless (fast)' if headless else 'visual'
    print(f'Starting training [{mode}] — close pygame window to stop.')
    print(f'{"Game":>6}  {"Score":>6}  {"Best":>6}  {"Mean":>8}  {"Roll100":>8}  {"Time":>8}')
    print('-' * 55)

    while True:
        if not headless:
            game.set_game_info(agent.n_games + 1, best_length)

        state_old = game.get_state()
        action = agent.get_action(state_old)

        reward, done, score = game.play_step(action)
        state_new = game.get_state()

        agent.train_short_memory(state_old, action, reward, state_new, done)
        agent.remember(state_old, action, reward, state_new, done)

        if done:
            final_length = len(game.snake)
            best_length = max(best_length, final_length)

            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            agent.trainer.sync_target()

            if score > best_score:
                best_score = score
                agent.model.save('model_best_score.pth')

            total_score += score
            mean_score = total_score / agent.n_games
            scores.append(score)
            mean_scores.append(mean_score)
            rolling_means = rolling_mean(scores, ROLLING_WINDOW)
            current_rolling = rolling_means[-1]
            elapsed = time.time() - start_time

            if current_rolling > best_rolling:
                best_rolling = current_rolling
                agent.model.save('model.pth')

            print(
                f'{agent.n_games:>6}  {score:>6}  {best_score:>6}  '
                f'{mean_score:>8.2f}  {current_rolling:>8.2f}  '
                f'{format_elapsed(elapsed):>8}'
            )

            if agent.n_games % plot_every == 0:
                plot(scores, mean_scores, rolling_means)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Snake RL agent')
    parser.add_argument('--visual', action='store_true')
    parser.add_argument('--plot-every', type=int, default=25)
    args = parser.parse_args()
    train(headless=not args.visual, plot_every=args.plot_every)