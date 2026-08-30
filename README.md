<div align="center">

```
███╗   ███╗██╗ ██████╗██████╗  ██████╗ ████████╗██╗   ██╗████████╗ ██████╗ ██████╗
████╗ ████║██║██╔════╝██╔══██╗██╔═══██╗╚══██╔══╝██║   ██║╚══██╔══╝██╔═══██╗██╔══██╗
██╔████╔██║██║██║     ██████╔╝██║   ██║   ██║   ██║   ██║   ██║   ██║   ██║██████╔╝
██║╚██╔╝██║██║██║     ██╔══██╗██║   ██║   ██║   ██║   ██║   ██║   ██║   ██║██╔══██╗
██║ ╚═╝ ██║██║╚██████╗██║  ██║╚██████╔╝   ██║   ╚██████╔╝   ██║   ╚██████╔╝██║  ██║
╚═╝     ╚═╝╚═╝ ╚═════╝╚═╝  ╚═╝ ╚═════╝   ╚═╝    ╚═════╝    ╚═╝    ╚═════╝ ╚═╝  ╚═╝
```

# 🤖 AI-Powered Micro-Programming Tutor

**An agentic LLM system that teaches code one step at a time - and learns from itself**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-black?style=for-the-badge&logo=ollama&logoColor=white)](https://ollama.ai)
[![Supabase](https://img.shields.io/badge/Supabase-Database-3ECF8E?style=for-the-badge&logo=supabase&logoColor=white)](https://supabase.com)
[![LeetCode](https://img.shields.io/badge/LeetCode-500_Problems-FFA116?style=for-the-badge&logo=leetcode&logoColor=white)](https://leetcode.com)
[![Penn State](https://img.shields.io/badge/Penn_State-CMPSC_496-1E407C?style=for-the-badge)](https://psu.edu)

*CMPSC 496 Independent Research · Penn State University · Spring 2026*

---

</div>

## 🧠 What Is This?

Most AI tools just **give you the answer**. This one doesn't.

`MicroTutor` is an agentic AI tutoring system that breaks programming problems into **ordered micro-steps**, evaluates each step independently, provides targeted hints on failure, and simulates multiple AI students at different skill levels - all running **completely offline** on your machine.

No OpenAI. No internet. No cheating.

```
YOU GIVE IT A PROBLEM
        ↓
LLM DECOMPOSES INTO 7-10 MICRO-STEPS
        ↓
THREE AI STUDENTS ANSWER EACH STEP
   [WEAK 0.5b]  [NORMAL 1.5b]  [STRONG 7b]
        ↓
EVALUATOR GRADES EACH ANSWER AGAINST RUBRIC
        ↓
HINT → RETRY → REVEAL (2 attempts max)
        ↓
FULL SOLUTION RECONSTRUCTED + SCORED
        ↓
EVERYTHING LOGGED TO SUPABASE
```

---

## ⚡ The Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR (qwen2.5:7b)                │
│                                                             │
│   ┌──────────────┐         ┌──────────────────────────┐    │
│   │  DECOMPOSER  │────────▶│       MICRO-STEPS        │    │
│   │  qwen2.5:7b  │         │  Step 1: signature       │    │
│   └──────────────┘         │  Step 2: initialize      │    │
│                             │  Step 3: loop header     │    │
│   ┌──────────────┐         │  ...                     │    │
│   │  EVALUATOR   │◀────────│  Step N: return          │    │
│   │  temp = 0.0  │         └──────────────────────────┘    │
│   │  rubric-     │                    ▲                     │
│   │  grounded    │                    │                     │
│   └──────────────┘         ┌──────────────────────┐        │
│          │                  │   THREE AI STUDENTS  │        │
│          │                  │                      │        │
│          │                  │ 🔴 WEAK   qwen2.5:0.5b│       │
│          │                  │    T=0.8  (lots of   │        │
│          │                  │           mistakes)  │        │
│          │                  │                      │        │
│          │                  │ 🟡 NORMAL qwen2.5:1.5b│       │
│          │                  │    T=0.5  (sometimes │        │
│          │                  │           wrong)     │        │
│          │                  │                      │        │
│          │                  │ 🟢 STRONG qwen2.5:7b │        │
│          │                  │    T=0.3  (usually   │        │
│          │                  │           correct)   │        │
│          │                  └──────────────────────┘        │
│          ▼                                                   │
│   ┌──────────────────────────────────────────────────┐      │
│   │          FEEDBACK LOOP                           │      │
│   │  Attempt 1 → grade → hint → Attempt 2 → grade   │      │
│   │  If still wrong → reveal correct answer          │      │
│   └──────────────────────────────────────────────────┘      │
│          ▼                                                   │
│   ┌──────────────────────────────────────────────────┐      │
│   │   SOLUTION RECONSTRUCTOR + SCORER                │      │
│   │   Assembles step answers → compares to ground    │      │
│   │   truth → score 0.0 to 1.0 → saved to Supabase  │      │
│   └──────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────┘
                              ▼
                    ┌─────────────────┐
                    │    SUPABASE     │
                    │  ┌───────────┐  │
                    │  │ problems  │  │
                    │  │ steps     │  │
                    │  │interactions│ │
                    │  └───────────┘  │
                    └─────────────────┘
```

---

## 🗃️ Database Schema

```sql
problems - 100 LeetCode problems with ground truth solutions
    id · slug · title · difficulty · description · topic_tags · solution

steps - decomposed micro-steps per problem
    id · problem_id · step_number · prompt · expected_type · rubric

interactions - every agent answer, grade, hint, and final score
    id · step_id · agent_level · attempt · answer · correct
       · hint_shown · final_answer · score · created_at
```

---

## 🛠️ Tech Stack

| Layer | Tech |
|-------|------|
| Language | Python 3.11+ |
| LLM Runtime | Ollama (fully local) |
| Models | Qwen2.5 - 0.5b, 1.5b, 7b |
| Database | Supabase (PostgreSQL) |
| Validation | Pydantic |
| Problems | LeetCode GraphQL API (500 problems) |
| Version Control | Git / GitHub |

---

## 🚀 Getting Started

### 1. Prerequisites

```bash
# Install Ollama
brew install ollama        # macOS
ollama serve               # start the server

# Pull all three student models
ollama pull qwen2.5:0.5b-instruct
ollama pull qwen2.5:1.5b-instruct
ollama pull qwen2.5:7b-instruct
```

### 2. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/CMPSC496.git
cd CMPSC496

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

> **Run everything through `.venv`** - use `.venv/bin/python` (or activate the venv
> first), never a bare/global `python`. A global interpreter is missing `supabase`
> and will fail on import.

### 3. Environment Setup

Create a `.env` file in the root:

```env
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_KEY=your-public-anon-key-here
```

### 4. Set Up Supabase

Run this in your Supabase SQL Editor:

```sql
create extension if not exists "pgcrypto";

create table problems (
  id uuid primary key default gen_random_uuid(),
  slug text unique not null,
  title text not null,
  difficulty text check (difficulty in ('Easy','Medium','Hard')),
  description text,
  topic_tags text[],
  source text default 'leetcode',
  solution text,
  created_at timestamptz default now()
);

create table steps (
  id uuid primary key default gen_random_uuid(),
  problem_id uuid references problems(id) on delete cascade,
  step_number int not null,
  prompt text not null,
  expected_type text check (expected_type in ('code','string')),
  rubric text,
  created_at timestamptz default now()
);

create table interactions (
  id uuid primary key default gen_random_uuid(),
  step_id uuid references steps(id) on delete cascade,
  agent_level text check (agent_level in ('weak','normal','strong')),
  attempt int check (attempt in (1,2)),
  answer text,
  correct boolean,
  hint_shown text,
  final_answer text,
  score float,
  created_at timestamptz default now()
);
```

### 5. Load Problems

```bash
# Fetch 500 LeetCode problems → problems_raw.json
python fetch_problems.py

# Upload to Supabase
python upload_to_supabase.py

# Generate ground truth solutions via LLM
python fetch_solutions.py
```

### 6. Run the Tutor

```bash
python run_phase1.py
```

---

## 📁 Project Structure

```
microprog-ai-tutor/
│
├── run_phase1.py          # 🧠 Main orchestrator - decompose, evaluate, score
├── student_agent.py       # 🎓 Three AI students (weak / normal / strong)
├── prompts.py             # 📝 System prompts for decomposer and evaluator
├── schemas.py             # 🔷 Pydantic models (StepItem, EvalResult, etc.)
├── ollama_client.py       # 🔌 HTTP client for local Ollama server
├── fetch_problems.py      # 📥 LeetCode problem scraper
├── upload_to_supabase.py  # ☁️  Bulk uploader to Supabase
├── fetch_solutions.py     # 🔑 Ground truth solution generator
├── .env                   # 🔐 Supabase credentials (never committed)
└── .gitignore
```

---

## 🧩 Key Engineering Decisions

### Why local LLMs?
Zero latency cost, no API keys, no data leaving the machine. Critical for a research system running 500 × 10 steps × 3 agents = **15,000+ LLM calls**.

### Why rubric-grounded evaluation?
Early versions let the evaluator reason freely - it hallucinated wrong reasons and invented incorrect reference answers. Passing the rubric directly into every eval call reduced false negatives by ~18%.

### Why three agents?
A single agent at one temperature doesn't represent the full student ability distribution. Three models at different parameter sizes and temperatures simulate realistic variation in beginner, intermediate, and advanced learners.

### Why reconstruct before scoring?
Individual step answers (single lines) look nothing like a complete function. Scoring `max_length = 0` against a full solution always returns 0. The reconstructor assembles all step answers into a coherent function first.

### Why reset context per question?
Early tests showed `validated_context` carrying hailstone sequence facts into dictionary inversion problems - causing the evaluator to hallucinate wrong reasons. Per-question context isolation eliminated cross-contamination entirely.

---

## 📊 Sample Analytics Queries

Once interaction data is collected, run these in Supabase SQL Editor:

```sql
-- Pass rate by agent level
SELECT agent_level,
       ROUND(AVG(correct::int) * 100, 1) AS pass_rate_pct,
       COUNT(*) AS total_attempts
FROM interactions
GROUP BY agent_level
ORDER BY pass_rate_pct DESC;

-- Average score by difficulty
SELECT p.difficulty,
       ROUND(AVG(i.score)::numeric, 2) AS avg_score,
       COUNT(DISTINCT p.id) AS problems
FROM interactions i
JOIN steps s ON i.step_id = s.id
JOIN problems p ON s.problem_id = p.id
WHERE i.score IS NOT NULL
GROUP BY p.difficulty
ORDER BY avg_score DESC;

-- Hardest problems (lowest pass rate)
SELECT p.title, p.difficulty,
       ROUND(AVG(i.correct::int) * 100, 1) AS pass_rate_pct
FROM interactions i
JOIN steps s ON i.step_id = s.id
JOIN problems p ON s.problem_id = p.id
GROUP BY p.title, p.difficulty
ORDER BY pass_rate_pct ASC
LIMIT 10;

-- Steps most often requiring a hint
SELECT s.prompt,
       COUNT(*) AS hint_count
FROM interactions i
JOIN steps s ON i.step_id = s.id
WHERE i.hint_shown != '' AND i.attempt = 1 AND i.correct = false
GROUP BY s.prompt
ORDER BY hint_count DESC
LIMIT 10;
```

---

## 🔬 Research Context

This project is part of **CMPSC 496 Independent Research** at Penn State University, investigating:

- **RQ1**: Can LLMs reliably decompose programming problems into ordered, gradable micro-steps?
- **RQ2**: How accurately can rubric-grounded prompting evaluate single-line student answers?
- **RQ3**: Does a multi-ability-level agent simulation reveal meaningful patterns in student error types?

---

## 🗺️ Roadmap

- [x] Phase 1 - Core tutoring loop with 3 agents
- [x] Supabase integration - problems, steps, interactions
- [x] LeetCode problem ingestion (500 problems)
- [x] Ground truth solution generation
- [x] Final answer reconstruction + scoring
- [ ] Phase 2 - Web UI for human student interaction
- [ ] Code execution sandbox for deterministic grading
- [ ] Analytics dashboard
- [ ] Fine-tune small model on collected interaction data

---

## 👤 Author

**Daksh Mainee**
B.S. Computer Science & AI · Penn State University
dzm6085@psu.edu

---

<div align="center">

*Built with 🧠 local LLMs, ☕ caffeine, and a firm belief that AI should teach - not just answer.*

</div>
