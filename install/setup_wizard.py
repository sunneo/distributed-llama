#!/usr/bin/env python3
import argparse
import html
import json
import os
import shlex
import shutil
import subprocess
import sys
from http.server import SimpleHTTPRequestHandler
from pathlib import Path
from socketserver import TCPServer
from typing import Dict, List
from urllib.parse import urlsplit

WIZARD_STEPS: List[Dict[str, object]] = [
    {
        "id": "node",
        "title": "Node Deployment",
        "diagram": "[MASTER: this node] ---- ethernet ---- [SLAVE NODES]",
        "description": "The current machine is usually the master node. Add slave nodes by IP/host.",
        "fields": [
            ("master_host", "Master host", "127.0.0.1"),
            ("worker_port", "Worker port", "9999"),
            ("slave_nodes", "Slave nodes (comma separated ssh_target[:worker_port])", ""),
            ("remote_deploy_dir", "Remote deploy dir", "/opt/distributed-llama"),
        ],
    },
    {
        "id": "model",
        "title": "Model Download",
        "diagram": "[HuggingFace/Local] -> models/<name>/dllama_model_*.m",
        "description": "Choose model preset and optionally trigger launch.py for guided model download.",
        "fields": [
            ("model_name", "Model preset (for launch.py)", "llama3_1_8b_instruct_q40"),
            ("model_path", "Model path", "models/llama3_1_8b_instruct_q40/dllama_model_llama3_1_8b_instruct_q40.m"),
            ("tokenizer_path", "Tokenizer path", "models/llama3_1_8b_instruct_q40/dllama_tokenizer_llama3_1_8b_instruct_q40.t"),
            ("remote_model_path", "Remote model path (optional)", ""),
        ],
    },
    {
        "id": "endpoint",
        "title": "OpenAI Endpoint",
        "diagram": "[Client] -> [LiteLLM bridge] -> [distributed-llama endpoint]",
        "description": "Configure OpenAI-compatible endpoint parameters and optional LiteLLM startup command.",
        "fields": [
            ("dllama_api_base", "Distributed-Llama API base", "http://127.0.0.1:9990/v1"),
            ("litellm_port", "LiteLLM bind port", "4000"),
            ("api_key", "API key for clients", "dllama-local-key"),
        ],
    },
    {
        "id": "chat_ui",
        "title": "Chat UI",
        "diagram": "[Browser UI] <-> [OpenAI-compatible endpoint]",
        "description": "Generate a local chat UI with file upload, new chat, and chat switching.",
        "fields": [
            ("chat_ui_port", "Local chat UI port", "7860"),
            ("chat_ui_file", "Chat UI HTML file", "chat_ui.html"),
        ],
    },
]


def default_settings() -> Dict[str, object]:
    return {
        "wizard_version": 1,
        "current_step": 0,
        "completed_steps": [],
        "config": {},
    }


def load_settings(path: Path) -> Dict[str, object]:
    if not path.exists():
        return default_settings()
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    base = default_settings()
    base.update(data if isinstance(data, dict) else {})
    if not isinstance(base.get("completed_steps"), list):
        base["completed_steps"] = []
    if not isinstance(base.get("config"), dict):
        base["config"] = {}
    base["current_step"] = max(0, min(int(base.get("current_step", 0)), len(WIZARD_STEPS) - 1))
    return base


def save_settings(path: Path, settings: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2, ensure_ascii=False)


class WizardState:
    def __init__(self, settings_path: Path):
        self.settings_path = settings_path
        self.settings = load_settings(settings_path)

    @property
    def current_step_index(self) -> int:
        return int(self.settings["current_step"])

    @property
    def current_step(self) -> Dict[str, object]:
        return WIZARD_STEPS[self.current_step_index]

    def config_value(self, key: str, default: str) -> str:
        return str(self.settings["config"].get(key, default))

    def set_config(self, key: str, value: str) -> None:
        self.settings["config"][key] = value

    def mark_complete(self, step_id: str) -> None:
        completed = self.settings["completed_steps"]
        if step_id not in completed:
            completed.append(step_id)

    def next_step(self) -> None:
        step_id = str(self.current_step["id"])
        self.mark_complete(step_id)
        self.settings["current_step"] = min(self.current_step_index + 1, len(WIZARD_STEPS) - 1)

    def previous_step(self) -> None:
        self.settings["current_step"] = max(self.current_step_index - 1, 0)

    def goto_step(self, index: int) -> None:
        self.settings["current_step"] = max(0, min(index, len(WIZARD_STEPS) - 1))

    def persist(self) -> None:
        save_settings(self.settings_path, self.settings)


def build_litellm_command(config: Dict[str, object]) -> str:
    model_name = str(config.get("model_name", "llama3_1_8b_instruct_q40"))
    dllama_api_base = str(config.get("dllama_api_base", "http://127.0.0.1:9990/v1"))
    litellm_port = str(config.get("litellm_port", "4000"))
    api_key = str(config.get("api_key", "dllama-local-key"))
    return (
        f"litellm --model {model_name} --api_base {dllama_api_base} "
        f"--port {litellm_port} --api_key {api_key}"
    )


def parse_slave_nodes(config: Dict[str, object]) -> List[Dict[str, str]]:
    default_worker_port = str(config.get("worker_port", "9999"))
    raw_nodes = str(config.get("slave_nodes", ""))
    parsed: List[Dict[str, str]] = []
    for raw_node in raw_nodes.split(","):
        node = raw_node.strip()
        if not node:
            continue

        ssh_target = node
        runtime_host = node.split("@", 1)[-1]
        runtime_port = default_worker_port

        if ":" in node:
            maybe_target, maybe_port = node.rsplit(":", 1)
            if maybe_port.isdigit():
                ssh_target = maybe_target
                runtime_host = maybe_target.split("@", 1)[-1]
                runtime_port = maybe_port

        parsed.append(
            {
                "ssh_target": ssh_target,
                "worker_address": f"{runtime_host}:{runtime_port}",
            }
        )
    return parsed


def build_mix_worker_command(config: Dict[str, object]) -> str:
    remote_deploy_dir = str(config.get("remote_deploy_dir", "/opt/distributed-llama"))
    remote_python = str(config.get("remote_python", "python3"))
    master_host = str(config.get("master_host", "127.0.0.1"))
    worker_port = str(config.get("worker_port", "9999"))
    remote_model_path = str(config.get("remote_model_path") or config.get("model_path", ""))
    remote_log_path = str(config.get("remote_worker_log", "/tmp/distributed-llama-worker.log"))

    command = [
        "cd",
        shlex.quote(f"{remote_deploy_dir}/mix/target"),
        "&&",
        "nohup",
        shlex.quote(remote_python),
        "-m",
        "distributed_llama_python.worker",
        "--host",
        shlex.quote(master_host),
        "--port",
        shlex.quote(worker_port),
    ]
    if remote_model_path:
        command.extend(["--model", shlex.quote(remote_model_path)])
    command.extend([">", shlex.quote(remote_log_path), "2>&1", "<", "/dev/null", "&"])
    return " ".join(command)


def build_mix_api_command(config: Dict[str, object], repo_root: Path) -> List[str]:
    api_base = str(config.get("dllama_api_base", "http://127.0.0.1:9990/v1"))
    parsed_api_base = urlsplit(api_base)
    api_host = parsed_api_base.hostname or str(config.get("master_host", "127.0.0.1"))
    api_port = str(parsed_api_base.port or 9990)
    model_path = Path(str(config.get("model_path", "")))
    tokenizer_path = Path(str(config.get("tokenizer_path", "")))
    if not model_path.is_absolute():
        model_path = repo_root / model_path
    if not tokenizer_path.is_absolute():
        tokenizer_path = repo_root / tokenizer_path

    command = [
        str(repo_root / "dllama-api"),
        "--host",
        api_host,
        "--port",
        api_port,
        "--model",
        str(model_path),
        "--tokenizer",
        str(tokenizer_path),
    ]

    workers = [node["worker_address"] for node in parse_slave_nodes(config)]
    if workers:
        command.extend(["--workers", *workers])
    return command


def write_nodes_file(settings_path: Path, config: Dict[str, object]) -> Path:
    nodes_path = settings_path.with_name("wizard-nodes.txt")
    ssh_targets = [node["ssh_target"] for node in parse_slave_nodes(config)]
    nodes_path.write_text("\n".join(ssh_targets) + "\n", encoding="utf-8")
    return nodes_path


def deploy_and_start_workers(state: WizardState, repo_root: Path) -> bool:
    nodes = parse_slave_nodes(state.settings["config"])
    if not nodes:
        print("⚠️ No slave nodes configured. Fill in 'slave_nodes' first.")
        return False

    config = state.settings["config"]
    nodes_path = write_nodes_file(state.settings_path, config)
    remote_dir = str(config.get("remote_deploy_dir", "/opt/distributed-llama"))

    package_result = subprocess.run(["bash", str(repo_root / "install/package_worker.sh")], cwd=repo_root, check=False)
    if package_result.returncode != 0:
        print("❌ Failed to package mixed worker bundle.")
        return False

    deploy_result = subprocess.run(
        ["bash", str(repo_root / "install/deploy_workers.sh"), str(nodes_path), remote_dir],
        cwd=repo_root,
        check=False,
    )
    if deploy_result.returncode != 0:
        print("❌ Failed to deploy mixed worker bundle.")
        return False

    remote_command = build_mix_worker_command(config)
    for node in nodes:
        print(f"▶ Starting mixed worker on {node['ssh_target']}")
        start_result = subprocess.run(["ssh", node["ssh_target"], remote_command], check=False)
        if start_result.returncode != 0:
            print(f"❌ Failed to start mixed worker on {node['ssh_target']}")
            return False

    print(f"✅ Deployed and started mixed workers using {nodes_path}")
    return True


def run_mix_api(state: WizardState, repo_root: Path) -> bool:
    config = state.settings["config"]
    model_path = Path(str(config.get("model_path", "")))
    tokenizer_path = Path(str(config.get("tokenizer_path", "")))
    if not model_path.is_absolute():
        model_path = repo_root / model_path
    if not tokenizer_path.is_absolute():
        tokenizer_path = repo_root / tokenizer_path

    if not config.get("model_path") or not model_path.exists():
        print(f"⚠️ Model file not found: {model_path}")
        print("   Use 'run-download' or edit 'model_path' before running mix.")
        return False

    if not config.get("tokenizer_path") or not tokenizer_path.exists():
        print(f"⚠️ Tokenizer file not found: {tokenizer_path}")
        print("   Use 'run-download' or edit 'tokenizer_path' before running mix.")
        return False

    binary_path = repo_root / "dllama-api"
    if not binary_path.exists():
        build_result = subprocess.run(["make", "dllama-api"], cwd=repo_root, check=False)
        if build_result.returncode != 0 or not binary_path.exists():
            print("❌ Failed to build dllama-api.")
            return False

    run_result = subprocess.run(build_mix_api_command(config, repo_root), cwd=repo_root, check=False)
    if run_result.returncode != 0:
        print("❌ Mix API exited with a non-zero status.")
        return False
    return True


def render_chat_ui_html(config: Dict[str, object]) -> str:
    api_base = html.escape(str(config.get("dllama_api_base", "http://127.0.0.1:9990/v1")), quote=True)
    return f"""<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <title>Distributed-Llama Chat UI</title>
  <style>
    body {{ font-family: sans-serif; margin: 0; display: flex; height: 100vh; }}
    #sidebar {{ width: 260px; border-right: 1px solid #ddd; padding: 12px; box-sizing: border-box; }}
    #main {{ flex: 1; display: flex; flex-direction: column; }}
    #messages {{ flex: 1; overflow: auto; padding: 12px; white-space: pre-wrap; }}
    #controls {{ padding: 12px; border-top: 1px solid #ddd; display: grid; gap: 8px; }}
    button {{ cursor: pointer; }}
    textarea {{ min-height: 80px; }}
    .msg-user {{ color: #005; }}
    .msg-assistant {{ color: #050; }}
  </style>
</head>
<body>
  <div id=\"sidebar\">
    <h3>Conversations</h3>
    <button onclick=\"newConversation()\">+ New conversation</button>
    <div id=\"conversation-list\"></div>
    <hr/>
    <label>API Base</label><input id=\"api-base\" value=\"{api_base}\" />
    <label>API Key</label><input id=\"api-key\" value=\"\" placeholder=\"Enter API key\" />
  </div>
  <div id=\"main\">
    <div id=\"messages\"></div>
    <div id=\"controls\">
      <input type=\"file\" id=\"file-input\" />
      <textarea id=\"prompt\" placeholder=\"Type your message\"></textarea>
      <button onclick=\"sendMessage()\">Send</button>
    </div>
  </div>
<script>
let conversations = [{{name: 'Conversation 1', messages: []}}];
let active = 0;
let uploadedContent = '';

function renderConversations() {{
  const list = document.getElementById('conversation-list');
  list.innerHTML = '';
  conversations.forEach((c, i) => {{
    const b = document.createElement('button');
    b.textContent = c.name;
    b.style.display = 'block';
    b.style.marginTop = '8px';
    b.onclick = () => {{ active = i; renderMessages(); renderConversations(); }};
    if (i === active) b.style.fontWeight = 'bold';
    list.appendChild(b);
  }});
}}

function renderMessages() {{
  const box = document.getElementById('messages');
  box.innerHTML = conversations[active].messages
    .map(m => `<div class=\"msg-${{m.role}}\"><b>${{m.role}}:</b> ${{m.content}}</div>`)
    .join('<hr/>');
  box.scrollTop = box.scrollHeight;
}}

function newConversation() {{
  conversations.push({{name: `Conversation ${{conversations.length + 1}}`, messages: []}});
  active = conversations.length - 1;
  renderConversations();
  renderMessages();
}}

document.getElementById('file-input').addEventListener('change', async (e) => {{
  const file = e.target.files[0];
  if (!file) return;
  uploadedContent = await file.text();
}});

async function sendMessage() {{
  const prompt = document.getElementById('prompt').value.trim();
  if (!prompt) return;
  const merged = uploadedContent ? `${{prompt}}\\n\\n[Uploaded file content]\\n${{uploadedContent.slice(0, 6000)}}` : prompt;
  const conv = conversations[active];
  conv.messages.push({{role: 'user', content: prompt}});
  renderMessages();

  const apiBase = document.getElementById('api-base').value.trim();
  const apiKey = document.getElementById('api-key').value.trim();
  const payload = {{ model: 'distributed-llama', messages: [...conv.messages.filter(m => m.role !== 'assistant').map(m => ({{role:m.role, content:m.content}}))] }};

  try {{
    const res = await fetch(`${{apiBase}}/chat/completions`, {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json', 'Authorization': 'ApiKey ' + apiKey }},
      body: JSON.stringify(payload)
    }});
    const data = await res.json();
    const assistant = data.choices?.[0]?.message?.content || JSON.stringify(data);
    conv.messages.push({{role:'assistant', content: assistant}});
  }} catch (e) {{
    conv.messages.push({{role:'assistant', content: `Request failed: ${{e}}`}});
  }}

  uploadedContent = '';
  document.getElementById('file-input').value = '';
  document.getElementById('prompt').value = '';
  renderMessages();
}}

renderConversations();
renderMessages();
</script>
</body>
</html>
"""


def start_chat_ui_server(html_path: Path, port: int) -> None:
    os.chdir(str(html_path.parent))
    handler = SimpleHTTPRequestHandler
    with TCPServer(("0.0.0.0", port), handler) as httpd:
        print(f"🌐 Chat UI available at http://127.0.0.1:{port}/{html_path.name}")
        print("Press Ctrl+C to stop chat UI server")
        httpd.serve_forever()


def try_start_litellm(config: Dict[str, object]) -> bool:
    if shutil.which("litellm") is None:
        print("⚠️ litellm command not found. Install with: pip install litellm")
        return False
    command = build_litellm_command(config)
    print("▶ Starting LiteLLM bridge process")
    subprocess.Popen(command.split(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return True


def run_step_editor(state: WizardState) -> None:
    step = state.current_step
    print("\n" + "=" * 70)
    print(f"Step {state.current_step_index + 1}/{len(WIZARD_STEPS)}: {step['title']}")
    print("=" * 70)
    print(step["diagram"])
    print(step["description"])
    print("\nCurrent values:")
    for key, label, default in step["fields"]:
        print(f"- {label}: {state.config_value(key, default)}")

    print("\nCommands: edit | next | prev | goto <n> | save | run-download | run-deploy | run-mix | run-litellm | run-chat-ui | quit")


def handle_command(command: str, state: WizardState, repo_root: Path) -> bool:
    parts = command.strip().split()
    action = parts[0] if parts else ""

    if action == "edit":
        for key, label, default in state.current_step["fields"]:
            current = state.config_value(key, default)
            value = input(f"{label} [{current}]: ").strip()
            if value:
                state.set_config(key, value)
        state.persist()
    elif action == "next":
        state.next_step()
        state.persist()
    elif action == "prev":
        state.previous_step()
        state.persist()
    elif action == "goto" and len(parts) > 1 and parts[1].isdigit():
        state.goto_step(int(parts[1]) - 1)
        state.persist()
    elif action == "save":
        state.persist()
        print(f"✅ Saved: {state.settings_path}")
    elif action == "run-download":
        model_name = str(state.settings["config"].get("model_name", "llama3_1_8b_instruct_q40"))
        subprocess.run([sys.executable, str(repo_root / "launch.py"), model_name], check=False)
    elif action == "run-deploy":
        deploy_and_start_workers(state, repo_root)
    elif action == "run-mix":
        run_mix_api(state, repo_root)
    elif action == "run-litellm":
        try_start_litellm(state.settings["config"])
    elif action == "run-chat-ui":
        chat_ui_file = Path(str(state.settings["config"].get("chat_ui_file", "chat_ui.html")))
        if not chat_ui_file.is_absolute():
            chat_ui_file = repo_root / chat_ui_file
        chat_ui_file.write_text(render_chat_ui_html(state.settings["config"]), encoding="utf-8")
        port = int(state.settings["config"].get("chat_ui_port", "7860"))
        start_chat_ui_server(chat_ui_file, port)
    elif action == "quit":
        state.persist()
        return False
    else:
        print("Unknown command.")
    return True


def run_tui(state: WizardState, repo_root: Path) -> None:
    print(f"🧭 settings.json path: {state.settings_path}")
    print("Resume supported: wizard stores current step and config after every action.")
    running = True
    while running:
        run_step_editor(state)
        running = handle_command(input("wizard> "), state, repo_root)


def run_gui(state: WizardState) -> None:
    import tkinter as tk

    root = tk.Tk()
    root.title("Distributed-Llama Setup Wizard")
    title_var = tk.StringVar()
    diagram_var = tk.StringVar()
    desc_var = tk.StringVar()
    status_var = tk.StringVar()
    entries = {}

    frame = tk.Frame(root, padx=12, pady=12)
    frame.pack(fill=tk.BOTH, expand=True)

    tk.Label(frame, textvariable=title_var, font=("Arial", 16, "bold")).pack(anchor="w")
    tk.Label(frame, textvariable=diagram_var, justify=tk.LEFT, fg="#444").pack(anchor="w", pady=(6, 0))
    tk.Label(frame, textvariable=desc_var, wraplength=760, justify=tk.LEFT).pack(anchor="w", pady=(6, 12))
    fields_frame = tk.Frame(frame)
    fields_frame.pack(fill=tk.X)

    def render():
        for widget in fields_frame.winfo_children():
            widget.destroy()
        entries.clear()

        step = state.current_step
        title_var.set(f"Step {state.current_step_index + 1}/{len(WIZARD_STEPS)} - {step['title']}")
        diagram_var.set(str(step["diagram"]))
        desc_var.set(str(step["description"]))

        for key, label, default in step["fields"]:
            tk.Label(fields_frame, text=label).pack(anchor="w")
            v = tk.StringVar(value=state.config_value(key, default))
            entries[key] = v
            tk.Entry(fields_frame, textvariable=v, width=100).pack(fill=tk.X, pady=(0, 6))
        status_var.set(f"settings.json: {state.settings_path}")

    def save_current_fields():
        for key, var in entries.items():
            state.set_config(key, var.get().strip())
        state.persist()

    button_frame = tk.Frame(frame)
    button_frame.pack(fill=tk.X, pady=(8, 0))

    def on_prev():
        save_current_fields()
        state.previous_step()
        render()

    def on_next():
        save_current_fields()
        state.next_step()
        render()

    def on_save():
        save_current_fields()
        status_var.set(f"Saved at step {state.current_step_index + 1}")

    def on_deploy():
        save_current_fields()
        if deploy_and_start_workers(state, Path(__file__).resolve().parents[1]):
            status_var.set("Deployed mixed workers")
        else:
            status_var.set("Worker deployment failed")

    def on_run_mix():
        save_current_fields()
        if run_mix_api(state, Path(__file__).resolve().parents[1]):
            status_var.set("Mix API exited successfully")
        else:
            status_var.set("Mix API failed")

    tk.Button(button_frame, text="Previous", command=on_prev).pack(side=tk.LEFT)
    tk.Button(button_frame, text="Next", command=on_next).pack(side=tk.LEFT, padx=8)
    tk.Button(button_frame, text="Save", command=on_save).pack(side=tk.LEFT)
    tk.Button(button_frame, text="Deploy Workers", command=on_deploy).pack(side=tk.LEFT, padx=8)
    tk.Button(button_frame, text="Start Mix API", command=on_run_mix).pack(side=tk.LEFT)
    tk.Label(frame, textvariable=status_var, fg="#555").pack(anchor="w", pady=(8, 0))

    render()
    root.mainloop()


def main() -> int:
    parser = argparse.ArgumentParser(description="Distributed-Llama setup wizard (TUI + GUI)")
    parser.add_argument("--mode", choices=["tui", "gui"], default="tui")
    parser.add_argument("--settings", default="settings.json")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    settings_path = Path(args.settings)
    if not settings_path.is_absolute():
        settings_path = repo_root / settings_path

    state = WizardState(settings_path)

    if args.mode == "gui":
        run_gui(state)
    else:
        run_tui(state, repo_root)

    return 0


if __name__ == "__main__":
    sys.exit(main())
