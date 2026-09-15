# ToolAgent — An AI Agent with Tool Use

An AI agent that decides, on its own, when to call a tool instead of just
answering from memory: a calculator, a Wikipedia lookup, a live weather
check, and the current date/time. Every tool call is shown in the UI so
you can see exactly what the agent did and why.

**How it works (for your resume/interview talking points):**
1. `tools.py` — defines each tool as a plain Python function, plus a JSON
   schema describing its name, purpose, and parameters (the format the
   LLM needs to know a tool exists and how to call it).
2. `agent.py` — the agentic loop: send the conversation + tool schemas to
   the model, check if it asked to call a tool, execute that tool in
   Python, feed the result back as a new message, and repeat. This is the
   core pattern behind every "AI agent" you'll hear about — reasoning and
   acting in a loop, not a single prompt-response.
3. `app.py` — Streamlit chat UI, with an expandable panel showing every
   tool call and its raw result underneath each answer.

This project deliberately uses **no ML libraries** — no embeddings, no
vector database, nothing with compiled native extensions beyond what
`requests` needs. Tool-calling is purely an LLM API feature, so the whole
thing runs on `groq` + `requests` + `streamlit`.

---

## 1. Run it locally

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Free Groq API key: https://console.groq.com
cp .env.example .env
# open .env and paste your key in place of your_groq_api_key_here

streamlit run app.py
```

Try asking:
- "What's 18% of 2450, and what day is it today?" (two tools in one turn)
- "What's the weather in Pune right now?"
- "Who was Ada Lovelace?"

---

## 2. Put it on GitHub

```bash
git init
git add .
echo ".env" >> .gitignore
echo "venv/" >> .gitignore
git commit -m "ToolAgent: an AI agent with calculator, Wikipedia, and weather tools"
git branch -M main
git remote add origin https://github.com/<your-username>/toolagent.git
git push -u origin main
```

## 3. Get a live shareable link (free)

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. **New app** → select your `toolagent` repo and `app.py` as the entry point.
3. Under **Advanced settings → Secrets**, add:
   ```
   GROQ_API_KEY = "your_actual_key"
   ```
4. Deploy. You'll get a public `.streamlit.app` link to put on your resume.

---

## Notes / extending it

- Add a new tool by writing a function in `tools.py`, adding its schema to
  `TOOLS_SCHEMA`, and registering it in `_DISPATCH` — `agent.py` needs no
  changes at all. That separation is worth mentioning in interviews.
- `max_steps` in `run_agent()` caps how many tool-call rounds happen
  before the agent is forced to answer — prevents infinite loops if the
  model keeps requesting tools.
- The Wikipedia and Open-Meteo APIs are both free and require no API key,
  so the only credential you need for this whole project is Groq's.
