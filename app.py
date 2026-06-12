"""Vercel ASGI entrypoint — landing page and project info API."""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI(title="Snake RL")

REPO_URL = "https://github.com/nilabh25bcs10235-beep/snake.rl"

LANDING_HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Snake RL — Pure ML DQN</title>
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
    h3 {{ margin: 24px 0 8px; color: var(--accent); }}
    p, li {{ color: var(--muted); line-height: 1.6; }}
    a {{ color: var(--accent); }}
    .badge {{
      display: inline-block;
      margin: 12px 0;
      padding: 8px 12px;
      border-radius: 8px;
      background: rgba(72, 220, 160, 0.12);
      border: 1px solid rgba(72, 220, 160, 0.35);
      color: var(--head);
      font-size: 14px;
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
  </style>
</head>
<body>
  <div class="card">
    <h1>Snake RL</h1>
    <p>Pure self-learning DQN agent for Snake. No hints — every move is learned by the neural network.</p>
    <div class="badge">Live on Vercel · Play &amp; train locally with pygame</div>

    <h3>Run on your machine</h3>
    <p>The pygame game cannot run inside Vercel serverless. Clone the repo and run locally:</p>
    <pre>git clone {REPO_URL}.git
cd snake.rl
python -m venv venv
venv\\Scripts\\activate
pip install -r requirements-ml.txt
python serve.py</pre>
    <p>Then open <a href="http://localhost:8080">http://localhost:8080</a> for the local launcher.</p>

    <h3>Train the agent</h3>
    <pre>python train.py --games 500
python train.py --target-score 55</pre>

    <h3>Watch the AI play</h3>
    <pre>python evaluate.py --visual
python evaluate.py --model ./model/model_best_score.pth --games 10</pre>

    <h3>Links</h3>
    <ul>
      <li><a href="{REPO_URL}">GitHub repository</a></li>
      <li><a href="/api/status">API status</a> (<code>/api/status</code>)</li>
    </ul>
  </div>
</body>
</html>"""


@app.get("/")
def home():
    return HTMLResponse(LANDING_HTML)


@app.get("/api/status")
def status():
    return {
        "name": "Snake RL",
        "ok": True,
        "framework": "FastAPI",
        "repo": REPO_URL,
        "play_locally": "pip install -r requirements-ml.txt && python evaluate.py --visual",
        "local_launcher": "python serve.py  →  http://localhost:8080",
        "note": "Pygame and PyTorch run on your machine, not on Vercel serverless.",
    }