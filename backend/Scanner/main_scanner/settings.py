"""
Scanner settings — single source of truth for the main scanner.

EDIT THIS FILE to change:
  - what sources are scanned (RSS feeds, NewsAPI queries, scrape targets, arXiv queries)
  - which AI keywords are matched
  - request timing, page sizes, output filename, log level

Everything the scanner needs to launch lives here. The scanner/ folder can be
copied to another project as-is along with requirements.txt.

Source curation principles (per KickstartAI feedback):
  - Focus on NL / EU enterprise AI adoption, regulation, implementation
  - Dutch credible news (NOS, Telegraaf, NL Times, MTsprout, Computable)
  - EU policy + research institutes (EC, Rijksoverheid, EURACTIV, TNO, NLAIC)
  - Consultancy and academic reports (McKinsey, BCG, Capgemini, Stanford HAI)
  - Selective international tech press: TechCrunch, MIT Technology Review
  - Avoid: Google News redirects (consent walls), generic consumer tech,
    press-release wires, opinion / hype pieces without enterprise angle
"""

from __future__ import annotations

import os

# Auto-load API keys from main_scanner/.env (sibling of this file).
# Safe no-op if python-dotenv isn't installed or .env doesn't exist.
try:
    from pathlib import Path as _Path
    from dotenv import load_dotenv
    load_dotenv(_Path(__file__).resolve().parent / ".env")
except ImportError:
    pass


# ===========================================================================
# RUNTIME TUNABLES
# ===========================================================================

# --- API keys -------------------------------------------------------------
# Key lives in main_scanner/.env (git-ignored), loaded above. Empty if unset,
# in which case the orchestrator skips NewsAPI sources.
NEWSAPI_KEY: str = os.environ.get("NEWSAPI_KEY", "")

# --- NewsAPI defaults -----------------------------------------------------
NEWSAPI_LANGUAGE: str = "en"
NEWSAPI_PAGE_SIZE: int = 100    # max allowed by NewsAPI
NEWSAPI_DAYS_BACK: int = 7      # how many days back to search

# --- HTTP defaults --------------------------------------------------------
REQUEST_DELAY: float = 1.2      # per-fetch politeness pause (now a per-worker throttle)
REQUEST_TIMEOUT: int = 20       # seconds
MAX_WORKERS: int = 8            # concurrent article fetches in NewsAPI/RSS sources
                                # (set to 1 for fully sequential fetching)

REQUEST_HEADERS: dict = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; KickstartAI-Scanner/1.0; "
        "+https://www.kickstart.ai)"
    )
}

# --- Output ---------------------------------------------------------------
OUTPUT_FILE: str = "data/scans/scanner_output.json"
LOG_LEVEL: str = "INFO"

# --- Excluded sources ------------------------------------------------------
# Articles from these sources are dropped from the final scanner output.
# Applied by both the main scanner (orchestrator) and the annotation scanner.
EXCLUDED_SOURCES: set = {
    # Press-release wires (no editorial filter)
    "GlobeNewswire",
    "PRNewswire",
    "PR Newswire",
    "PR Newswire UK",
    "Business Wire",
    "Accesswire",
    # Low-signal programming / coin sites
    "Pypi.org",
    "C-sharpcorner.com",
    "Crypto Briefing",
    # Per KickstartAI feedback: NL-based but no NL focus
    "The Next Web",
    # Geographic / audience focus mismatch
    "The Times of India",
    "BusinessLine",
    "Slashdot.org",
}


# ===========================================================================
# PAYWALLED / SUBSCRIPTION SOURCES
# ===========================================================================
# Domains that require a subscription. The scanner does NOT extract the full
# body from these — it keeps only the freely-visible preview (the article lede,
# capped at PAYWALL_PREVIEW_CHARS) or, if that is too thin, the licensed snippet
# the news API provides. No paywall is ever circumvented. Free/open sources are
# unaffected (they still get full-text extraction). Matched by URL domain.
PAYWALLED_SOURCES: set = {
    "nrc.nl", "telegraaf.nl", "fd.nl", "volkskrant.nl", "parool.nl", "trouw.nl",
    "sifted.eu", "ft.com", "wsj.com", "economist.com", "bloomberg.com",
    "theinformation.com", "nytimes.com", "washingtonpost.com",
}

# For paywalled sources, keep at most this many characters of the free preview
# (the publicly-visible lede). A cap, not a minimum — shorter previews are kept
# whole; longer extractions (leaky client-side paywalls) are trimmed to this.
PAYWALL_PREVIEW_CHARS: int = 600


# ===========================================================================
# AI KEYWORDS — used by the relevance filter (any match qualifies)
# ===========================================================================

AI_KEYWORDS_EN: list = [
    # --- Core ---
    "AI",
    "artificial intelligence",
    "generative AI",
    "large language model",
    "LLM",
    "foundation model",
    "machine learning",
    "deep learning",
    "neural network",
    "natural language processing",
    "NLP",
    "GPT",
    "ChatGPT",
    "transformer model",
    "diffusion model",
    "autonomous systems",
    # --- Regulation / governance / ethics ---
    "AI Act",
    "EU AI Act",
    "AI regulation",
    "responsible AI",
    "trustworthy AI",
    "AI governance",
    "AI safety",
    "AI ethics",
    "responsible deployment",
    "responsible applied AI",
    # --- Geographic focus ---
    "Dutch AI",
    "European AI",
    "AI Europe",
    "AI Netherlands",
    "AI sovereignty",
    "digital strategy",
    "digital sovereignty",
    # --- Enterprise / adoption / implementation ---
    "enterprise AI",
    "AI adoption",
    "AI implementation",
    "AI deployment",
    "AI pilot",
    "AI scaling",
    "AI use case",
    "AI in enterprise",
    "AI in organizations",
    "AI in business",
    "applied AI",
    "AI capabilities",
    "AI talent",
    "AI ecosystem",
    # --- KickstartAI use-case categories ---
    "agentic AI",
    "AI agents",
    "AI copilot",
    "AI assistant",
    "predictive diagnostics",
    "predictive maintenance",
    "digital twin",
    # --- Sector verticals (KickstartAI partner industries) ---
    "AI in healthcare",
    "AI in medicine",
    "clinical AI",
    "AI in finance",
    "AI in banking",
    "AI in retail",
    "AI in aviation",
    "AI in transport",
    "AI in logistics",
    "AI in agriculture",
    "AI in education",
    "AI in public sector",
    "AI in government",
]

AI_KEYWORDS_NL: list = [
    # --- Core ---
    "kunstmatige intelligentie",
    "algoritme",
    "algoritmes",
    "generatieve AI",
    "taalmodel",
    "machinaal leren",
    "deep learning",
    "slimme technologie",
    # --- Regulation / governance / ethics ---
    "AI-wet",
    "AI-regulering",
    "AI-beleid",
    "verantwoordelijke AI",
    "verantwoorde AI",
    "betrouwbare AI",
    "digitale strategie",
    "digitale soevereiniteit",
    # --- Government / public sector ---
    "digitale overheid",
    "digitalisering",
    "innovatie",
    "data science",
    # --- Enterprise / adoption / implementation ---
    "AI in bedrijven",
    "AI-adoptie",
    "AI in het bedrijfsleven",
    "AI Nederland",
    "toegepaste AI",
    "AI capaciteiten",
    "AI ecosysteem",
    "AI talent",
    "AI-pilot",
    "AI-implementatie",
    # --- Use-case categories ---
    "agentic AI",
    "voorspellend onderhoud",
    "digitale tweelingen",
    "klinische AI",
    # --- Sector verticals ---
    "AI in de zorg",
    "AI in de gezondheidszorg",
    "AI in de financiële sector",
    "AI in retail",
    "AI in vervoer",
    "AI in mobiliteit",
    "AI in landbouw",
    "AI in onderwijs",
    "AI in overheid",
    # --- Themes ---
    "AI voor klimaat",
    "AI voor duurzaamheid",
]


# ===========================================================================
# NEWSAPI QUERIES
# ===========================================================================

NEWSAPI_SOURCES: dict = {
    # Focused queries: prioritise NL / EU / enterprise adoption over generic AI hype
    "queries_en": [
        "EU AI Act",
        "AI regulation Europe",
        "AI governance",
        "AI adoption enterprise",
        "responsible AI",
        "AI strategy Netherlands",
        "AI implementation enterprise",
        "AI in organizations",
        "European AI policy",
        "AI sovereignty Europe",
    ],
    "queries_nl": [
        "kunstmatige intelligentie bedrijven",
        "AI adoptie Nederland",
        "AI in de zorg",
        "generatieve AI bedrijfsleven",
        "AI beleid Nederland",
    ],
    "domain_queries": [
        # Enterprise-angled tech press
        {
            "query": "AI adoption OR AI enterprise OR AI implementation",
            "domains": "techcrunch.com",
            "language": "en",
            "tag": "news",
        },
        {
            "query": "AI enterprise OR AI adoption OR AI implementation",
            "domains": "technologyreview.com",
            "language": "en",
            "tag": "news",
        },
        # Dutch news (combined domains)
        {
            "query": "kunstmatige intelligentie OR AI OR algoritme",
            "domains": "nos.nl,telegraaf.nl,nltimes.nl,fd.nl,mtsprout.nl,computable.nl",
            "language": "nl",
            "tag": "dutch_news",
        },
        # EU policy / regulation
        {
            "query": "AI Act OR AI regulation OR digital strategy OR AI governance",
            "domains": "euractiv.com,politico.eu",
            "language": "en",
            "tag": "policy_regulation",
        },
        # Tier-1 international with EU/enterprise AI coverage
        {
            "query": "AI adoption OR AI regulation OR AI Europe",
            "domains": "reuters.com,bbc.com,theguardian.com,ft.com",
            "language": "en",
            "tag": "news",
        },
        # Enterprise / strategy research outlets
        {
            "query": "AI strategy OR AI enterprise OR responsible AI OR AI adoption",
            "domains": "hbr.org,mit.edu,technologyreview.com",
            "language": "en",
            "tag": "research_reports",
        },
        # --- Sector-specific queries (KickstartAI partner verticals) ---
        # Restricted to already-approved domains so no new sites are introduced
        {
            "query": "AI healthcare OR AI in medicine OR clinical AI OR predictive diagnostics",
            "domains": "nos.nl,telegraaf.nl,nltimes.nl,fd.nl,mtsprout.nl,computable.nl,technologyreview.com",
            "language": "en",
            "tag": "news",
        },
        {
            "query": "AI banking OR AI in finance OR financial AI adoption",
            "domains": "fd.nl,hbr.org,ft.com,technologyreview.com,reuters.com",
            "language": "en",
            "tag": "news",
        },
        {
            "query": "AI infrastructure OR predictive maintenance OR digital twin OR smart grid",
            "domains": "euractiv.com,politico.eu,technologyreview.com,reuters.com",
            "language": "en",
            "tag": "news",
        },
        {
            "query": "AI sustainability OR AI climate OR AI for circular economy OR AI agriculture",
            "domains": "euractiv.com,theguardian.com,technologyreview.com,reuters.com",
            "language": "en",
            "tag": "news",
        },
        {
            "query": "AI talent OR AI ecosystem OR applied AI OR AI capabilities",
            "domains": "hbr.org,mit.edu,technologyreview.com,mtsprout.nl,computable.nl",
            "language": "en",
            "tag": "research_reports",
        },
        {
            "query": "AI agents OR agentic AI OR AI copilot OR AI assistant enterprise",
            "domains": "techcrunch.com,technologyreview.com,hbr.org",
            "language": "en",
            "tag": "news",
        },
    ],
}


# ===========================================================================
# RSS FEEDS — KickstartAI-suggested NL news + EU/research/ecosystem
# ===========================================================================

RSS_FEEDS: list = [
    # --- International tech press (selective: enterprise-angled only) ---
    {
        "url": "https://techcrunch.com/category/artificial-intelligence/feed/",
        "tag": "news",
        "source": "TechCrunch AI",
    },
    {
        "url": "https://www.technologyreview.com/feed/",
        "tag": "news",
        "source": "MIT Technology Review",
    },
    # --- Dutch credible news (per KickstartAI list) — tagged dutch_news ---
    {
        "url": "https://feeds.nos.nl/nosnieuwsalgemeen",
        "tag": "dutch_news",
        "source": "NOS Nieuws",
    },
    {
        "url": "https://feeds.nos.nl/nosnieuwstech",
        "tag": "dutch_news",
        "source": "NOS Tech",
    },
    {
        "url": "https://www.telegraaf.nl/rss",
        "tag": "dutch_news",
        "source": "Telegraaf",
    },
    {
        "url": "https://nltimes.nl/rss.xml",
        "tag": "dutch_news",
        "source": "NL Times",
    },
    {
        "url": "https://www.mtsprout.nl/feed",
        "tag": "dutch_news",
        "source": "MTsprout",
    },
    {
        "url": "https://www.computable.nl/rss",
        "tag": "dutch_news",
        "source": "Computable",
    },
    # --- Policy / Regulation (EU + NL government) ---
    {
        "url": "https://digital-strategy.ec.europa.eu/en/rss.xml",
        "tag": "policy_regulation",
        "source": "European Commission Digital Strategy",
    },
    {
        "url": "https://www.rijksoverheid.nl/onderwerpen/kunstmatige-intelligentie/rss",
        "tag": "policy_regulation",
        "source": "Rijksoverheid - Kunstmatige Intelligentie",
    },
    {
        "url": "https://www.rijksoverheid.nl/onderwerpen/digitalisering/rss",
        "tag": "policy_regulation",
        "source": "Rijksoverheid - Digitalisering",
    },
    {
        "url": "https://www.euractiv.com/sections/digital/feed/",
        "tag": "policy_regulation",
        "source": "EURACTIV Digital",
    },
    # --- Research / Reports ---
    {
        "url": "https://hai.stanford.edu/news/rss.xml",
        "tag": "research_reports",
        "source": "Stanford HAI",
    },
    {
        "url": "https://aiindex.stanford.edu/rss/",
        "tag": "research_reports",
        "source": "Stanford AI Index",
    },
    # --- Dutch Ecosystem ---
    {
        "url": "https://www.tno.nl/en/newsroom/rss/",
        "tag": "dutch_ecosystem",
        "source": "TNO",
    },
    # --- Additional EU enterprise / policy / digital-rights ---
    {
        "url": "https://sifted.eu/feed",
        "tag": "news",
        "source": "Sifted",
    },
    {
        "url": "https://www.politico.eu/section/technology/feed/",
        "tag": "policy_regulation",
        "source": "Politico EU - Technology",
    },
    {
        "url": "https://algorithmwatch.org/en/feed/",
        "tag": "policy_regulation",
        "source": "Algorithm Watch",
    },
    {
        "url": "https://www.bitsoffreedom.nl/feed/",
        "tag": "policy_regulation",
        "source": "Bits of Freedom",
    },
]


# ===========================================================================
# WEB SCRAPE TARGETS (listing pages with link filters)
# ===========================================================================

SCRAPE_TARGETS: list = [
    {
        "type": "listing",
        "url": "https://eur-lex.europa.eu/search.html?scope=EURLEX&text=artificial+intelligence+act&lang=en&searchType=quick",
        "tag": "policy_regulation",
        "source": "EUR-Lex",
        "link_filter": "eur-lex.europa.eu/legal-content",
        "max_articles": 10,
    },
    {
        "type": "listing",
        "url": "https://www.mckinsey.com/capabilities/quantumblack/our-insights",
        "tag": "research_reports",
        "source": "McKinsey AI",
        "link_filter": "/capabilities/quantumblack/our-insights/",
        "max_articles": 10,
    },
    {
        "type": "listing",
        "url": "https://www.bcg.com/capabilities/artificial-intelligence/insights",
        "tag": "research_reports",
        "source": "BCG AI",
        "link_filter": "/publications/",
        "max_articles": 10,
    },
    {
        "type": "listing",
        "url": "https://hai.stanford.edu/news",
        "tag": "research_reports",
        "source": "Stanford HAI",
        "link_filter": "/news/",
        "max_articles": 10,
    },
    {
        "type": "listing",
        "url": "https://www.capgemini.com/insights/research-library/",
        "tag": "research_reports",
        "source": "Capgemini Research",
        "link_filter": "/insights/research-library/",
        "max_articles": 10,
    },
    {
        "type": "listing",
        "url": "https://www.tno.nl/en/focus-areas/digitalization/roadmaps/data-intelligence/",
        "tag": "dutch_ecosystem",
        "source": "TNO",
        "link_filter": "tno.nl/en/",
        "max_articles": 10,
    },
    {
        "type": "listing",
        "url": "https://www.deloitte.com/nl/nl/pages/deloitte-digital/articles/",
        "tag": "dutch_ecosystem",
        "source": "Deloitte NL",
        "link_filter": "deloitte.com/nl/",
        "max_articles": 8,
    },
]


# ===========================================================================
# arXiv queries — applied / enterprise focus
# ===========================================================================

ARXIV_QUERIES: list = [
    "large language model enterprise applications",
    "generative AI business",
    "AI regulation policy governance",
    "responsible AI safety alignment",
    "foundation model fine-tuning",
    "AI adoption organizations",
    "AI ethics fairness accountability",
    "AI implementation healthcare",
    # KickstartAI-aligned applied / deployment / sector themes
    "AI deployment enterprise healthcare",
    "AI agents enterprise applications",
    "digital twin infrastructure AI",
    "AI predictive maintenance industrial",
]

ARXIV_MAX_RESULTS: int = 20
