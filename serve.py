"""Local web launcher for Snake RL. Opens http://localhost:8080 in your browser."""

import json
import os
import subprocess
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable
HOST = '127.0.0.1'
PORT = 8080
BASE_URL = f'http://localhost:{PORT}'


def model_exists():
    return os.path.isfile(os.path.join(PROJECT_DIR, 'model', 'model.pth'))


def _quote(arg):
    return f'"{arg}"' if ' ' in arg else arg


def _cmd_line(command):
    return ' '.join([_quote(PYTHON)] + [_quote(a) for a in command])


def spawn_app(command, title='Snake RL'):
    """Launch a pygame app in its own window (no terminal)."""
    cmd_line = _cmd_line(command)
    if sys.platform == 'win32':
        subprocess.Popen(
            f'start "{title}" {cmd_line}',
            cwd=PROJECT_DIR,
            shell=True,
        )
    else:
        subprocess.Popen([PYTHON] + command, cwd=PROJECT_DIR)


PAGE = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Snake RL Launcher</title>
  <style>
    :root {{
      --bg: #0c1220;
      --panel: #121a2a;
      --accent: #64b4ff;
      --head: #48dc9f;
      --text: #e8eef8;
      --muted: #8fa0b8;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      font-family: "Segoe UI", system-ui, sans-serif;
      background: radial-gradient(circle at top, #182238, var(--bg) 55%);
      color: var(--text);
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 24px;
    }}
    .card {{
      width: min(720px, 100%);
      background: var(--panel);
      border: 1px solid #243048;
      border-radius: 16px;
      padding: 32px;
      box-shadow: 0 20px 60px rgba(0, 0, 0, 0.35);
    }}
    h1 {{ margin: 0 0 8px; color: var(--head); }}
    p {{ color: var(--muted); line-height: 1.6; }}
    .url {{
      display: inline-block;
      margin: 12px 0 24px;
      padding: 8px 12px;
      border-radius: 8px;
      background: #0b111c;
      color: var(--accent);
      text-decoration: none;
      font-family: Consolas, monospace;
    }}
    .actions {{
      display: grid;
      gap: 12px;
      margin: 24px 0;
    }}
    .train-row {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 10px;
    }}
    .section-label {{
      margin: 8px 0 4px;
      font-size: 13px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    button {{
      cursor: pointer;
      border: none;
      border-radius: 10px;
      padding: 14px 18px;
      font-size: 15px;
      font-weight: 600;
      color: #081018;
      background: linear-gradient(135deg, var(--head), #2ec88a);
    }}
    button.secondary {{
      color: var(--text);
      background: #1b2940;
      border: 1px solid #2d4266;
    }}
    pre {{
      background: #0b111c;
      border-radius: 10px;
      padding: 16px;
      overflow-x: auto;
      color: #c7d5ea;
      font-size: 13px;
      line-height: 1.5;
    }}
    #status {{
      min-height: 24px;
      margin-top: 12px;
      color: var(--head);
      font-weight: 600;
    }}
    .warn {{
      margin-top: 16px;
      padding: 12px 14px;
      border-radius: 10px;
      background: rgba(255, 170, 80, 0.12);
      border: 1px solid rgba(255, 170, 80, 0.35);
      color: #ffd9a8;
      display: none;
    }}
  </style>
</head>
<body>
  <div class="card">
    <h1>Snake RL</h1>
    <p>Pure self-learning DQN agent for Snake. <strong>Train</strong> opens a fast visual
    pygame window so you can watch the AI learn. <strong>Play</strong> runs the trained model.</p>
    <a class="url" href="{BASE_URL}">{BASE_URL}</a>

    <div class="actions">
      <div class="section-label">Train (visual, fast)</div>
      <div class="train-row">
        <button class="secondary" onclick="train(100)">100 games</button>
        <button class="secondary" onclick="train(300)">300 games</button>
        <button class="secondary" onclick="train(1000)">1000 games</button>
      </div>
      <div class="section-label">Evaluate</div>
      <button onclick="launch('play')">Play — watch trained AI</button>
    </div>

    <div id="status"></div>
    <div id="warn" class="warn">No trained model found yet. Run training first, then click Play.</div>

    <h3>Manual setup</h3>
    <pre>git clone https://github.com/nilabh25bcs10235-beep/snake.rl.git
cd snake.rl
python -m venv venv
venv\\Scripts\\activate
pip install -r requirements-ml.txt
python serve.py</pre>

    <h3>CLI commands</h3>
    <pre>python train.py --games 500
python evaluate.py --visual
python evaluate.py --model ./model/model_best_score.pth --games 10</pre>
  </div>

  <script>
    async function refreshStatus() {{
      const res = await fetch('/api/status');
      const data = await res.json();
      document.getElementById('warn').style.display = data.model_ready ? 'none' : 'block';
    }}

    async function launch(action) {{
      const status = document.getElementById('status');
      status.textContent = 'Launching...';
      const res = await fetch('/api/' + action, {{ method: 'POST' }});
      const data = await res.json();
      status.textContent = data.message;
    }}

    async function train(games) {{
      const status = document.getElementById('status');
      status.textContent = 'Starting training...';
      const res = await fetch('/api/train', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ games }}),
      }});
      const data = await res.json();
      status.textContent = data.message;
    }}

    refreshStatus();
  </script>
</body>
</html>"""


class LauncherHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def _send(self, code, body, content_type):
        data = body.encode('utf-8') if isinstance(body, str) else body
        self.send_response(code)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json(self, payload, code=200):
        self._send(code, json.dumps(payload), 'application/json')

    def do_GET(self):
        path = urlparse(self.path).path
        if path == '/':
            self._send(200, PAGE, 'text/html; charset=utf-8')
        elif path == '/api/status':
            self._json({
                'model_ready': model_exists(),
                'url': BASE_URL,
            })
        else:
            self.send_error(404)

    def _read_json_body(self):
        length = int(self.headers.get('Content-Length', 0))
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length))

    def do_POST(self):
        path = urlparse(self.path).path
        if path == '/api/play':
            if not model_exists():
                self._json({
                    'ok': False,
                    'message': 'No model found. Train first (100+ games).',
                }, 400)
                return
            spawn_app(['evaluate.py', '--visual'], title='Snake RL — Play')
            self._json({
                'ok': True,
                'message': 'Pygame window launching — check taskbar if hidden.',
            })
        elif path == '/api/train':
            body = self._read_json_body()
            games = int(body.get('games', 100))
            if games not in (100, 300, 1000):
                self._json({'ok': False, 'message': 'Choose 100, 300, or 1000 games.'}, 400)
                return
            spawn_app(
                ['train.py', '--visual', '--fast', '--quiet', '--games', str(games)],
                title=f'Snake RL — Train {games}',
            )
            self._json({
                'ok': True,
                'message': f'Training window opening ({games} games) — watch the snake learn!',
            })
        else:
            self.send_error(404)


def main():
    os.chdir(PROJECT_DIR)
    server = HTTPServer((HOST, PORT), LauncherHandler)
    print(f'Snake RL launcher running at {BASE_URL}')
    print('Press Ctrl+C to stop.')
    webbrowser.open(BASE_URL)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nStopped.')
        server.server_close()


if __name__ == '__main__':
    main()