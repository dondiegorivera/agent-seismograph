# Agent Seismograph Data Gatherer

Automated signal detection for the closed-loop AI agent transition.  
Uses **Tavily** (intelligent search) + **Groq** (fast LLM analysis).

## Quick Start

```bash
# 1. Clone
git clone https://github.com/youruser/agent-seismograph
cd agent-seismograph

# 2. Install
pip install -r requirements.txt

# 3. Configure
cp .env.example .env
# Edit .env with your API keys

# 4. Run
source .env  # or export the variables
python gatherer.py --schedule
```

## Architecture

```
┌─────────────────────────────────────────┐
│           TRIGGER (cron)                │
│  GitHub Actions / Hetzner Docker        │
└─────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│         SEARCH (Tavily API)             │
│  3 queries per prediction               │
│  ~5 results per query                   │
│  Deduped and combined                   │
└─────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│        ANALYZE (Groq + Llama 3.3)       │
│  Extract concrete signals               │
│  Score evidence strength/direction      │
│  Link to prediction IDs                 │
└─────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│             OUTPUT                      │
│  ./output/{date}-scan.json              │
│  ./output/latest.json                   │
│  (Optional) Push to R2/GitHub Pages     │
└─────────────────────────────────────────┘
```

## Cost Estimate

| Service | Free Tier | Our Usage | Monthly Cost |
|---------|-----------|-----------|--------------|
| Groq | 30 req/min, ~14K tokens/min | ~50 req/day | **$0** |
| Tavily | 1000 searches/month | ~40/day = 1200/mo | **$0-40** |
| GitHub Actions | 2000 min/month | ~5 min/day | **$0** |
| Hetzner CX22 | - | Alternative | **€4.5/mo** |

**Total: $0-45/month**

## Usage

### List available predictions
```bash
python gatherer.py --list
```

### Run today's scheduled scan
```bash
python gatherer.py --schedule
```

### Scan specific predictions
```bash
python gatherer.py --predictions INF-01 SEC-05 SWAN-06
```

### Scan all predictions
```bash
python gatherer.py --all
```

## Deployment Options

### Option A: GitHub Actions (Recommended)

1. Fork this repo
2. Add secrets in Settings > Secrets:
   - `GROQ_API_KEY`
   - `TAVILY_API_KEY`
3. Enable Actions
4. Runs daily at 10:00 UTC

### Option B: Hetzner Docker

```bash
cd agent-seismograph/docker
cp ../.env.example .env
# Edit .env with your keys
docker-compose up -d
```

## Output Format

```json
{
  "report_type": "daily_scan",
  "scan_date": "2025-02-05",
  "predictions_scanned": 6,
  "results": [
    {
      "prediction_id": "SEC-05",
      "analysis": {
        "signals": [
          {
            "title": "Malicious npm package targets AI agents",
            "evidence_strength": "strong",
            "evidence_direction": "confirming"
          }
        ],
        "overall_activity": "high"
      }
    }
  ]
}
```

## File Structure

```
agent-seismograph/
├── gatherer.py              # Main entry point
├── requirements.txt
├── .env.example
├── output/                  # Generated JSON files
├── .github/workflows/       # GitHub Actions
└── docker/                  # Hetzner deployment
```
