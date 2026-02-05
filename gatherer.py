#!/usr/bin/env python3
"""
Agent Seismograph Data Gatherer
Uses Tavily for intelligent search + Groq for analysis
"""

import os
import json
import asyncio
import argparse
import random
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import httpx

# ============================================================================
# CONFIGURATION
# ============================================================================

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
GROQ_MODEL = "openai/gpt-oss-120b"  # Fast, high quality
TAVILY_SEARCH_DEPTH = "advanced"  # or "basic" for faster/cheaper
GROQ_MIN_REQUEST_INTERVAL = float(os.environ.get("GROQ_MIN_REQUEST_INTERVAL", "2.5"))
GROQ_MAX_RETRIES = int(os.environ.get("GROQ_MAX_RETRIES", "4"))
GROQ_BACKOFF_BASE = float(os.environ.get("GROQ_BACKOFF_BASE", "2.0"))
GROQ_BACKOFF_MAX = float(os.environ.get("GROQ_BACKOFF_MAX", "30.0"))
GROQ_MAX_SOURCES = int(os.environ.get("GROQ_MAX_SOURCES", "6"))
GROQ_MAX_SOURCE_CHARS = int(os.environ.get("GROQ_MAX_SOURCE_CHARS", "700"))
GROQ_MAX_QUERY_SUMMARY_CHARS = int(os.environ.get("GROQ_MAX_QUERY_SUMMARY_CHARS", "300"))
GROQ_MAX_CONTEXT_CHARS = int(os.environ.get("GROQ_MAX_CONTEXT_CHARS", "8000"))
GROQ_MAX_OUTPUT_TOKENS = int(os.environ.get("GROQ_MAX_OUTPUT_TOKENS", "1200"))

BASE_DIR = Path(__file__).parent
CONFIG_DIR = BASE_DIR / "config"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
_GROQ_NEXT_REQUEST_TIME = 0.0

# ============================================================================
# PREDICTION CATALOG (legacy embedded; overridden by config/predictions.json)
# ============================================================================

PREDICTIONS = {
    "INF-01": {
        "id": "INF-01",
        "category": "infrastructure",
        "title": "Compute Denial of Service (CDoS)",
        "description": "Agent traffic causing bill shock bankruptcies",
        "search_queries": [
            "AI agent API costs bill shock startup 2025",
            "bot traffic serverless billing spike incident",
            "Vercel Cloudflare agent traffic costs"
        ],
        "trigger_keywords": ["bill shock", "bankrupted by bots", "agent traffic costs", "compute costs agent"]
    },
    "INF-02": {
        "id": "INF-02",
        "category": "infrastructure",
        "title": "API Re-monopolization / Agent Editions",
        "description": "SaaS launching Agent Tier APIs",
        "search_queries": [
            "API agent tier pricing announcement 2025",
            "Stripe Shopify agent API launch",
            "SaaS automation tier pricing"
        ],
        "trigger_keywords": ["agent tier", "agent API", "automation pricing", "agent edition"]
    },
    "INF-03": {
        "id": "INF-03",
        "category": "infrastructure",
        "title": "Rate Limit → Economic Limit",
        "description": "Shift to credits/prepaid wallets",
        "search_queries": [
            "API credits prepaid wallet launch 2025",
            "usage based billing API SaaS",
            "per-operation API pricing"
        ],
        "trigger_keywords": ["credits", "prepaid wallet", "per-operation", "usage-based API"]
    },
    "SEC-05": {
        "id": "SEC-05",
        "category": "security",
        "title": "Agent supply-chain risk",
        "description": "Backdoored agent pack scandal",
        "search_queries": [
            "malicious AI agent package NPM PyPI 2025",
            "LLM agent framework vulnerability CVE",
            "backdoored agent pack discovered"
        ],
        "trigger_keywords": ["malicious agent", "backdoor pack", "agent CVE", "supply chain attack"]
    },
    "SEC-06": {
        "id": "SEC-06",
        "category": "security",
        "title": "Prompt injection via documents",
        "description": "Invoice/PDF injection triggering actions",
        "search_queries": [
            "prompt injection PDF invoice attack 2025",
            "BEC prompt injection AI agent",
            "document injection LLM trigger"
        ],
        "trigger_keywords": ["prompt injection", "PDF attack", "invoice hijack", "document injection"]
    },
    "ATK-01": {
        "id": "ATK-01",
        "category": "attack",
        "title": "Operational Spam",
        "description": "Support queues flooded by agents",
        "search_queries": [
            "AI bot support ticket flooding 2025",
            "automated refund abuse spike",
            "Zendesk cost spike bot tickets"
        ],
        "trigger_keywords": ["support flood", "ticket spam", "refund abuse", "complaint bots"]
    },
    "ATK-05": {
        "id": "ATK-05",
        "category": "attack",
        "title": "Review/Identity Fraud Networks",
        "description": "Cross-platform synthetic personas",
        "search_queries": [
            "fake review network AI generated 2025",
            "synthetic identity fraud ring AI",
            "Amazon Trustpilot review manipulation"
        ],
        "trigger_keywords": ["fake review", "synthetic identity", "persona farm", "review fraud"]
    },
    "MKT-02": {
        "id": "MKT-02",
        "category": "market",
        "title": "Micro-Optimization Death Spirals",
        "description": "SEO/pricing gaming outpaces platforms",
        "search_queries": [
            "Google search quality AI spam 2025",
            "SEO AI content death spiral",
            "pricing algorithm war automation"
        ],
        "trigger_keywords": ["SEO death spiral", "content spam", "pricing war", "platform gaming"]
    },
    "MKT-03": {
        "id": "MKT-03",
        "category": "market",
        "title": "Clean Data Liquidity Crisis",
        "description": "Verified data feeds become premium",
        "search_queries": [
            "website blocks AI bot scraping 2025",
            "robots.txt AI crawler block",
            "verified data feed premium pricing"
        ],
        "trigger_keywords": ["AI bot block", "scraping banned", "verified data", "data paywall"]
    },
    "META-03": {
        "id": "META-03",
        "category": "meta",
        "title": "AgentOps / Observability",
        "description": "Tracing and debugging become mandatory",
        "search_queries": [
            "LangSmith agent observability 2025",
            "AI agent debugging tool launch",
            "AgentOps monitoring startup funding"
        ],
        "trigger_keywords": ["AgentOps", "agent observability", "agent tracing", "eval harness"]
    },
    "SWAN-06": {
        "id": "SWAN-06",
        "category": "black_swan",
        "title": "Backdoored Agent Pack",
        "description": "Popular pack exfiltrating across thousands",
        "search_queries": [
            "malicious AI agent pack discovered 2025",
            "backdoor agent framework thousands affected",
            "supply chain attack AI agents CVE"
        ],
        "trigger_keywords": ["malicious pack", "backdoor discovered", "widespread compromise"]
    },
    "BIZ-04": {
        "id": "BIZ-04",
        "category": "business",
        "title": "Civic & Bureaucratic DDoS",
        "description": "Government portals overwhelmed",
        "search_queries": [
            "FOIA flooding AI bots government 2025",
            "permit system overload automated",
            "government portal shutdown bot traffic"
        ],
        "trigger_keywords": ["FOIA flood", "permit overload", "government portal down"]
    }
}

# Load full prediction catalog from config (overrides embedded list)
CATALOG_PATH = CONFIG_DIR / "predictions.json"
if not CATALOG_PATH.exists():
    raise FileNotFoundError(f"Prediction catalog not found: {CATALOG_PATH}")
with open(CATALOG_PATH, "r") as f:
    PREDICTION_CATALOG = json.load(f)
_predictions = PREDICTION_CATALOG.get("predictions", [])
if not isinstance(_predictions, list) or not _predictions:
    raise ValueError(f"Prediction catalog missing or empty: {CATALOG_PATH}")
PREDICTIONS = {p["id"]: p for p in _predictions if "id" in p}

# Legacy fallback schedule: which predictions to scan on which days
SCHEDULE = {
    "daily": ["SEC-05", "SEC-06", "ATK-01", "SWAN-06"],
    "monday": ["INF-01", "INF-02", "INF-03", "MKT-02"],
    "wednesday": ["INF-01", "MKT-03", "ATK-05", "META-03"],
    "friday": ["INF-01", "INF-02", "BIZ-04", "MKT-02"],
}

# ============================================================================
# TAVILY SEARCH
# ============================================================================

async def tavily_search(query: str, max_results: int = 5) -> dict:
    """
    Search using Tavily API
    
    Tavily provides:
    - search: Basic search with snippets
    - search_context: Optimized for RAG, returns clean content
    - extract: Get full content from URLs
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            "https://api.tavily.com/search",
            json={
                "api_key": TAVILY_API_KEY,
                "query": query,
                "search_depth": TAVILY_SEARCH_DEPTH,
                "include_answer": True,
                "include_raw_content": False,
                "max_results": max_results,
                "include_domains": [],  # Optional: limit to specific domains
                "exclude_domains": ["reddit.com"],  # Often noisy
            }
        )
        response.raise_for_status()
        return response.json()


async def tavily_search_context(query: str, max_tokens: int = 4000) -> str:
    """
    Get search context optimized for LLM consumption
    Returns a single string with relevant content
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            "https://api.tavily.com/search",
            json={
                "api_key": TAVILY_API_KEY,
                "query": query,
                "search_depth": TAVILY_SEARCH_DEPTH,
                "include_answer": True,
                "include_raw_content": False,
                "max_results": 5,
            }
        )
        response.raise_for_status()
        data = response.json()
        
        # Build context string
        context_parts = []
        if data.get("answer"):
            context_parts.append(f"Summary: {data['answer']}\n")
        
        for i, result in enumerate(data.get("results", []), 1):
            context_parts.append(
                f"[{i}] {result.get('title', 'No title')}\n"
                f"URL: {result.get('url', '')}\n"
                f"Content: {result.get('content', '')}\n"
            )
        
        return "\n".join(context_parts)

# ============================================================================
# GROQ ANALYSIS
# ============================================================================

class GroqRateLimitError(RuntimeError):
    pass


def _parse_groq_duration_seconds(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    raw = value.strip().lower()
    try:
        return float(raw)
    except ValueError:
        pass

    # Groq docs use values such as "7.66s" and "2m59.56s"
    match = re.fullmatch(r"(?:(\d+)m)?\s*([0-9]+(?:\.[0-9]+)?)s", raw)
    if not match:
        return None

    minutes = float(match.group(1) or 0)
    seconds = float(match.group(2))
    return (minutes * 60.0) + seconds


def _groq_cooldown_from_headers(headers: httpx.Headers, fallback_seconds: float) -> float:
    header_delays = [
        _parse_groq_duration_seconds(headers.get("retry-after")),
        _parse_groq_duration_seconds(headers.get("x-ratelimit-reset-tokens")),
        _parse_groq_duration_seconds(headers.get("x-ratelimit-reset-requests")),
    ]
    valid_delays = [d for d in header_delays if d is not None and d >= 0]
    if valid_delays:
        return max(GROQ_MIN_REQUEST_INTERVAL, min(max(valid_delays), GROQ_BACKOFF_MAX))
    return max(GROQ_MIN_REQUEST_INTERVAL, min(fallback_seconds, GROQ_BACKOFF_MAX))


async def _respect_groq_cooldown():
    global _GROQ_NEXT_REQUEST_TIME
    now = time.monotonic()
    wait_seconds = _GROQ_NEXT_REQUEST_TIME - now
    if wait_seconds > 0:
        await asyncio.sleep(wait_seconds)
    _GROQ_NEXT_REQUEST_TIME = max(_GROQ_NEXT_REQUEST_TIME, time.monotonic()) + GROQ_MIN_REQUEST_INTERVAL


def _extend_groq_cooldown(delay_seconds: float):
    global _GROQ_NEXT_REQUEST_TIME
    _GROQ_NEXT_REQUEST_TIME = max(_GROQ_NEXT_REQUEST_TIME, time.monotonic() + max(0.0, delay_seconds))


def _analysis_failure_payload(exc: Exception) -> dict:
    message = str(exc)
    if isinstance(exc, GroqRateLimitError) or "429" in message or "Too Many Requests" in message:
        return {
            "signals": [],
            "overall_activity": "none",
            "summary": "Analysis skipped due to Groq rate limits; retry on the next run.",
        }
    return {
        "signals": [],
        "overall_activity": "none",
        "summary": "Analysis unavailable due to API or parsing error.",
    }


def _trim_text(value: str, max_chars: int) -> str:
    if max_chars <= 0 or len(value) <= max_chars:
        return value
    return value[:max_chars].rstrip() + "... [truncated]"


def _source_relevance_score(source: dict, prediction: dict) -> int:
    text = f"{source.get('title', '')} {source.get('content', '')}".lower()
    keywords = [k.lower() for k in prediction.get("trigger_keywords", []) if k]
    if not keywords:
        return 0
    return sum(1 for kw in keywords if kw in text)


async def groq_analyze(search_context: str, prediction: dict) -> dict:
    """
    Use Groq to analyze search results against a prediction
    """
    system_prompt = """You are an intelligence analyst for the "Agent Seismograph" system.
You detect early signals of AI agent ecosystem changes.

Your task: Analyze search results to find signals relevant to a specific prediction.

RULES:
1. Only extract CONCRETE signals (real events, announcements, incidents)
2. Ignore speculation, opinion pieces, or theoretical discussions
3. Score evidence_strength: "strong" (definitive), "moderate" (clear but partial), "weak" (early indicator)
4. Score evidence_direction: "confirming" (supports prediction), "contradicting" (undermines it), "neutral"
5. If no relevant signals found, return empty signals array

OUTPUT FORMAT (strict JSON):
{
  "signals": [
    {
      "title": "Brief headline",
      "description": "2-3 sentence summary",
      "source_url": "URL from search results",
      "source_name": "Publication name",
      "published_date": "YYYY-MM-DD or 'unknown'",
      "evidence_strength": "strong|moderate|weak",
      "evidence_direction": "confirming|contradicting|neutral",
      "relevance_rationale": "Why this relates to the prediction"
    }
  ],
  "overall_activity": "none|low|moderate|high",
  "summary": "1-2 sentence summary of findings"
}"""

    user_prompt = f"""PREDICTION TO TRACK:
ID: {prediction['id']}
Title: {prediction['title']}
Description: {prediction['description']}
Trigger Keywords: {', '.join(prediction.get('trigger_keywords', []))}

SEARCH RESULTS:
{search_context}

Analyze these results for signals relevant to the prediction. Return JSON only."""

    max_attempts = GROQ_MAX_RETRIES + 1
    fallback_backoff = GROQ_BACKOFF_BASE

    async with httpx.AsyncClient(timeout=60.0) as client:
        for attempt in range(1, max_attempts + 1):
            await _respect_groq_cooldown()
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": GROQ_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.2,
                    "max_tokens": GROQ_MAX_OUTPUT_TOKENS,
                    "response_format": {"type": "json_object"}
                }
            )

            if response.status_code == 429:
                delay = _groq_cooldown_from_headers(
                    response.headers,
                    fallback_backoff + random.uniform(0, 1),
                )
                _extend_groq_cooldown(delay)
                if attempt >= max_attempts:
                    raise GroqRateLimitError("Groq rate limit reached after retries")
                print(f"    Groq rate-limited, retrying in {delay:.1f}s (attempt {attempt}/{max_attempts})")
                await asyncio.sleep(delay)
                fallback_backoff = min(GROQ_BACKOFF_MAX, fallback_backoff * 2)
                continue

            response.raise_for_status()
            data = response.json()

            # Parse the JSON response
            content = data["choices"][0]["message"]["content"]
            return json.loads(content)

    raise RuntimeError("Groq analysis failed after retries")

# ============================================================================
# MAIN SCANNING LOGIC
# ============================================================================

async def scan_prediction(prediction_id: str) -> dict:
    """
    Scan a single prediction:
    1. Run all search queries
    2. Combine results
    3. Analyze with Groq
    """
    prediction = PREDICTIONS.get(prediction_id)
    if not prediction:
        raise ValueError(f"Unknown prediction: {prediction_id}")
    
    print(f"  Scanning {prediction_id}: {prediction['title']}")
    
    # Gather search results from all queries
    all_context = []
    all_sources = []
    
    for query in prediction.get("search_queries", []):
        try:
            result = await tavily_search(query, max_results=3)
            
            # Collect sources for deduplication
            for r in result.get("results", []):
                if r.get("url") not in [s.get("url") for s in all_sources]:
                    all_sources.append(r)
            
            # Build context
            if result.get("answer"):
                answer = _trim_text(str(result["answer"]), GROQ_MAX_QUERY_SUMMARY_CHARS)
                all_context.append(f"Query '{query}' summary: {answer}")
            
        except Exception as e:
            print(f"    Warning: Search failed for '{query}': {e}")
    
    # Format combined context
    context_parts = all_context.copy()
    ranked_sources = sorted(
        all_sources,
        key=lambda source: _source_relevance_score(source, prediction),
        reverse=True,
    )
    selected_sources = ranked_sources[:max(1, GROQ_MAX_SOURCES)]
    for i, source in enumerate(selected_sources, 1):
        source_content = _trim_text(str(source.get("content", "")), GROQ_MAX_SOURCE_CHARS)
        context_parts.append(
            f"[{i}] {source.get('title', 'No title')}\n"
            f"URL: {source.get('url', '')}\n"
            f"Content: {source_content}\n"
        )
    
    combined_context = _trim_text("\n\n".join(context_parts), GROQ_MAX_CONTEXT_CHARS)
    
    # Analyze with Groq
    try:
        analysis = await groq_analyze(combined_context, prediction)
    except Exception as e:
        print(f"    Warning: Analysis failed: {e}")
        analysis = _analysis_failure_payload(e)
    
    # Build output
    return {
        "prediction_id": prediction_id,
        "prediction_title": prediction["title"],
        "prediction_category": prediction["category"],
        "scan_timestamp": datetime.utcnow().isoformat() + "Z",
        "queries_run": len(prediction.get("search_queries", [])),
        "sources_found": len(all_sources),
        "analysis": analysis
    }


async def run_daily_scan(prediction_ids: list[str]) -> dict:
    """
    Run scans for a list of predictions
    """
    scan_date = datetime.utcnow().strftime("%Y-%m-%d")
    
    print(f"Agent Seismograph Daily Scan - {scan_date}")
    print(f"Scanning {len(prediction_ids)} predictions...")
    print()
    
    results = []
    for pred_id in prediction_ids:
        try:
            result = await scan_prediction(pred_id)
            results.append(result)
            
            # Rate limiting - be nice to APIs
            await asyncio.sleep(1)
            
        except Exception as e:
            print(f"  ERROR scanning {pred_id}: {e}")
            results.append({
                "prediction_id": pred_id,
                "error": str(e),
                "scan_timestamp": datetime.utcnow().isoformat() + "Z"
            })
    
    # Build daily report
    report = {
        "report_type": "daily_scan",
        "scan_date": scan_date,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "predictions_scanned": len(prediction_ids),
        "results": results,
        "summary": {
            "total_signals": sum(
                len(r.get("analysis", {}).get("signals", []))
                for r in results
            ),
            "high_activity": [
                r["prediction_id"] for r in results
                if r.get("analysis", {}).get("overall_activity") in ["moderate", "high"]
            ]
        }
    }
    
    return report


def _expand_schedule_items(items: list[str]) -> tuple[set[str], list[str]]:
    result: set[str] = set()
    unknown: list[str] = []
    for item in items:
        if item == "all":
            result.update(PREDICTIONS.keys())
        elif item == "all_swans":
            result.update(
                pid
                for pid, pred in PREDICTIONS.items()
                if pid.startswith("SWAN-") or pred.get("category") == "black_swan"
            )
        elif item in PREDICTIONS:
            result.add(item)
        else:
            unknown.append(item)
    return result, unknown


def _get_catalog_schedule_ids(today: str) -> list[str]:
    schedule = PREDICTION_CATALOG.get("category_schedule", {})
    if not isinstance(schedule, dict) or not schedule:
        return []

    items: list[str] = []
    items.extend(schedule.get("daily", []))

    if today in {"monday", "wednesday", "friday"}:
        items.extend(schedule.get("monday_wednesday_friday", []))
    if today in {"tuesday", "thursday"}:
        items.extend(schedule.get("tuesday_thursday", []))
    if today == "saturday":
        items.extend(schedule.get("saturday", []))
    if today == "sunday":
        items.extend(schedule.get("sunday", []))

    # Support direct day keys if present in the catalog
    items.extend(schedule.get(today, []))

    predictions, unknown = _expand_schedule_items(items)
    if unknown:
        print(f"Warning: Unknown schedule items ignored: {', '.join(sorted(set(unknown)))}")
    return list(predictions)


def get_todays_predictions() -> list[str]:
    """
    Get list of prediction IDs to scan today based on schedule
    """
    today = datetime.utcnow().strftime("%A").lower()

    catalog_ids = _get_catalog_schedule_ids(today)
    if catalog_ids:
        return catalog_ids

    predictions = set(SCHEDULE.get("daily", []))
    predictions.update(SCHEDULE.get(today, []))
    return list(predictions)


def save_report(report: dict, output_dir: Path = OUTPUT_DIR):
    """
    Save report to JSON file
    """
    scan_date = report.get("scan_date", datetime.utcnow().strftime("%Y-%m-%d"))
    filename = f"{scan_date}-scan.json"
    filepath = output_dir / filename
    
    with open(filepath, "w") as f:
        json.dump(report, f, indent=2)
    
    # Also save as latest.json
    latest_path = output_dir / "latest.json"
    with open(latest_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\nSaved: {filepath}")
    print(f"Saved: {latest_path}")
    
    return filepath

# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Agent Seismograph Data Gatherer")
    parser.add_argument(
        "--predictions", "-p",
        nargs="+",
        help="Specific prediction IDs to scan (e.g., INF-01 SEC-05)"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Scan all predictions"
    )
    parser.add_argument(
        "--schedule", "-s",
        action="store_true",
        help="Use today's schedule (default)"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available predictions"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory"
    )
    
    args = parser.parse_args()

    # List mode
    if args.list:
        print("Available Predictions:")
        print("-" * 60)
        for pid, pred in sorted(PREDICTIONS.items()):
            print(f"  {pid}: {pred['title']}")
        print()
        print("Today's schedule:", get_todays_predictions())
        return 0

    # Validate API keys
    if not GROQ_API_KEY:
        print("ERROR: GROQ_API_KEY environment variable not set")
        return 1
    if not TAVILY_API_KEY:
        print("ERROR: TAVILY_API_KEY environment variable not set")
        return 1
    
    # Determine which predictions to scan
    if args.predictions:
        prediction_ids = args.predictions
    elif args.all:
        prediction_ids = list(PREDICTIONS.keys())
    else:
        prediction_ids = get_todays_predictions()
    
    # Validate predictions
    for pid in prediction_ids:
        if pid not in PREDICTIONS:
            print(f"ERROR: Unknown prediction ID: {pid}")
            return 1
    
    # Run scan
    report = asyncio.run(run_daily_scan(prediction_ids))
    
    # Save output
    args.output.mkdir(exist_ok=True)
    save_report(report, args.output)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SCAN COMPLETE")
    print("=" * 60)
    print(f"Predictions scanned: {report['predictions_scanned']}")
    print(f"Total signals found: {report['summary']['total_signals']}")
    if report['summary']['high_activity']:
        print(f"High activity: {', '.join(report['summary']['high_activity'])}")
    
    return 0


if __name__ == "__main__":
    exit(main())
