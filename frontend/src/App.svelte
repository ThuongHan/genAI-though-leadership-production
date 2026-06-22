<script>
  // ── State ──────────────────────────────────────────────────────────────────
  let step = 'idle'
  // idle | scanning | articles | interpreting | interpreted |
  // generating | posts | post_action | feedback_input |
  // feedback_refining | auto_refining | auto_reviewed | final | error

  let logs            = []
  let scanPhase       = 'collecting'  // 'collecting' | 'scoring'
  let collectCount    = 0
  let scoreCount      = 0
  let scoreTotal      = 0
  let scanWarning     = ''            // non-empty when scanner fell back to old file
  let articles        = []
  let cachedFileMeta  = null
  let selectedArticle = null
  let interpretation  = null
  let posts           = []
  let selectedPost    = null
  let finalPost       = null
  let autoFinalPost   = null          // best post from auto-refine loop
  let refineStatus    = ''            // current human-readable refine step label
  let feedbackText    = ''
  let autoHistory     = []
  let errorMsg        = ''
  let copied          = false

  let interpretSubStep = 0
  let generateSubStep  = 0
  $: pipelineStage = (
    ['scanning', 'articles'].includes(step) ? 1 :
    ['interpreting', 'interpreted'].includes(step) ? 2 :
    ['generating', 'posts', 'post_action'].includes(step) ? 3 :
    ['feedback_input', 'feedback_refining', 'auto_refining', 'auto_reviewed'].includes(step) ? 4 :
    step === 'final' ? 5 : 0
  )

  // Matches Python log lines like "2026-06-15 17:22:52,819 INFO module message"
  const RAW_LOG_RE = /^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}/

  const DIMENSIONS = [
    'tone_of_voice', 'language_and_style', 'coherence_readability',
    'discourse_structure', 'specificity', 'historical_similarity',
  ]

  const DIMENSION_GUIDE = {
    tone_of_voice:          'Does it sound like an authentic human expert, not a robot?',
    language_and_style:     'Is the writing clear, direct, and suited to LinkedIn?',
    coherence_readability:  'Does it flow from one idea to the next without confusion?',
    discourse_structure:    'Are there any violations present? contrastive structure/from...to.../This/That...?',
    specificity:            'Does it include concrete facts, examples, or numbers?',
    historical_similarity:  'Does it match the style of successful KickstartAI posts?',
  }

  // ── SSE helper ─────────────────────────────────────────────────────────────
  async function readStream(response, onEvent) {
    const reader  = response.body.getReader()
    const decoder = new TextDecoder()
    let   buffer  = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() ?? ''
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const raw = line.slice(6).trim()
          if (raw) {
            try { onEvent(JSON.parse(raw)) } catch { /* ignore partial */ }
          }
        }
      }
    }
  }

  // ── 0. Load existing filter results ───────────────────────────────────────
  async function useCached() {
    try {
      const res = await fetch('/api/articles/cached')
      if (!res.ok) {
        const text = await res.text()
        let msg = `No saved results found (HTTP ${res.status})`
        try { msg = JSON.parse(text).detail || msg } catch { if (text.trim()) msg = text.slice(0, 200) }
        throw new Error(msg)
      }
      const data  = await res.json()
      cachedFileMeta = { source_file: data.source_file, age_hours: data.age_hours }
      articles = data.articles
      step     = 'articles'
    } catch (e) {
      errorMsg = String(e)
      step     = 'error'
    }
  }

  // ── 1. Scan ────────────────────────────────────────────────────────────────
  async function startScan() {
    step         = 'scanning'
    logs         = []
    scanWarning  = ''
    cachedFileMeta = null
    scanPhase    = 'collecting'
    collectCount = 0
    scoreCount   = 0
    scoreTotal   = 0
    try {
      const res = await fetch('/api/scan?force=true', { method: 'POST' })
      await readStream(res, (data) => {
        if (data.type === 'log' || data.type === 'progress') {
          const msg = data.message
          logs = [...logs, msg]

          if (msg.includes('⚠️')) {
            scanWarning = msg
          }

          const scored = msg.match(/scored\s+(\d+)\/(\d+)/)
          if (scored) {
            scanPhase  = 'scoring'
            scoreCount = +scored[1]
            scoreTotal = +scored[2]
          } else if (/Loaded\s+\d+\s+articles/.test(msg)) {
            scanPhase = 'scoring'
            const m = msg.match(/Loaded\s+(\d+)/)
            if (m) scoreTotal = +m[1]
          } else if (/Scoring articles|chain-of-thought|weighted_filter/.test(msg)) {
            scanPhase = 'scoring'
          } else if (/Total:\s+\d+\s+unique/.test(msg)) {
            const m = msg.match(/Total:\s+(\d+)\s+unique/)
            if (m) collectCount = +m[1]
          } else if (/step.*scan|Starting scanner|run_scanner/.test(msg)) {
            scanPhase = 'collecting'
          }

        } else if (data.type === 'done') {
          articles = data.articles
          step     = 'articles'
        } else if (data.type === 'error') {
          errorMsg = data.message
          step     = 'error'
        }
      })
    } catch (e) {
      errorMsg = String(e)
      step     = 'error'
    }
  }

  // ── 2. Interpret ───────────────────────────────────────────────────────────
  async function selectArticle(article) {
    selectedArticle  = article
    step             = 'interpreting'
    logs             = []
    interpretSubStep = 1
    const _t2 = setTimeout(() => { if (step === 'interpreting') interpretSubStep = 2 }, 4000)
    const _t3 = setTimeout(() => { if (step === 'interpreting') interpretSubStep = 3 }, 12000)
    try {
      const res = await fetch('/api/interpret', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({
          id:        article.id,
          title:     article.title,
          full_text: article.full_text,
          url:       article.url,
        }),
      })
      if (!res.ok) {
        const text = await res.text()
        let msg = `Interpretation failed (HTTP ${res.status})`
        try { msg = JSON.parse(text).detail || msg } catch { if (text.trim()) msg = text.slice(0, 300) }
        throw new Error(msg)
      }
      interpretation = await res.json()
      step           = 'interpreted'
    } catch (e) {
      errorMsg = String(e)
      step     = 'error'
    } finally {
      clearTimeout(_t2)
      clearTimeout(_t3)
    }
  }

  // ── 3. Generate ────────────────────────────────────────────────────────────
  async function generatePosts() {
    step            = 'generating'
    logs            = []
    generateSubStep = 1
    const _t2 = setTimeout(() => { if (step === 'generating') generateSubStep = 2 }, 3000)
    const _t3 = setTimeout(() => { if (step === 'generating') generateSubStep = 3 }, 8000)
    try {
      const res = await fetch('/api/generate', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({
          interpretation: interpretation.interpretation,
          metadata:       interpretation.metadata,
        }),
      })
      if (!res.ok) {
        const text = await res.text()
        let msg = `Generation failed (HTTP ${res.status})`
        try { msg = JSON.parse(text).detail || msg } catch { if (text.trim()) msg = text.slice(0, 300) }
        throw new Error(msg)
      }
      const data = await res.json()
      posts = data.posts
      step  = 'posts'
    } catch (e) {
      errorMsg = String(e)
      step     = 'error'
    } finally {
      clearTimeout(_t2)
      clearTimeout(_t3)
    }
  }

  // ── 4. Pick post & action ──────────────────────────────────────────────────
  function selectPost(post) {
    selectedPost = post
    step         = 'post_action'
  }

  function keepPost() {
    finalPost = selectedPost.content
    step      = 'final'
  }

  function startFeedback() {
    feedbackText = ''
    step         = 'feedback_input'
  }

  async function submitFeedback() {
    if (!feedbackText.trim()) return
    step = 'feedback_refining'
    try {
      const res = await fetch('/api/refine/feedback', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({
          post_content: selectedPost.content,
          feedback:     feedbackText,
        }),
      })
      if (!res.ok) throw new Error('Refinement failed')
      const data = await res.json()
      finalPost  = data.refined_post
      step       = 'final'
    } catch (e) {
      errorMsg = String(e)
      step     = 'error'
    }
  }

  // ── 5. Auto-refine ─────────────────────────────────────────────────────────
  async function startAutoRefine() {
    step          = 'auto_refining'
    logs          = []
    autoHistory   = []
    autoFinalPost = null
    refineStatus  = 'Starting…'
    try {
      const res = await fetch('/api/refine/auto', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ post_content: selectedPost.content }),
      })
      await readStream(res, (data) => {
        if (data.type === 'progress') {
          if (!RAW_LOG_RE.test(data.message)) {
            refineStatus = data.message
          }
        } else if (data.type === 'evaluation') {
          autoHistory = [...autoHistory, {
            iteration:   data.iteration,
            post:        data.post,
            evaluations: data.evaluations,
            avg:         data.avg,
          }]
        } else if (data.type === 'done') {
          autoFinalPost = data.final_post
          step          = 'auto_reviewed'   // show all results, let user pick
        } else if (data.type === 'error') {
          errorMsg = data.message
          step     = 'error'
        }
      })
    } catch (e) {
      errorMsg = String(e)
      step     = 'error'
    }
  }

  function useAutoPost(content) {
    finalPost = content
    step      = 'final'
  }

  // ── Helpers ────────────────────────────────────────────────────────────────
  function reset() {
    step            = 'idle'
    logs            = []
    scanWarning     = ''
    articles        = []
    cachedFileMeta  = null
    selectedArticle = null
    interpretation  = null
    posts           = []
    selectedPost    = null
    finalPost       = null
    autoFinalPost   = null
    refineStatus    = ''
    feedbackText    = ''
    autoHistory     = []
    errorMsg         = ''
    copied           = false
    interpretSubStep = 0
    generateSubStep  = 0
  }

  function copyPost() {
    navigator.clipboard.writeText(finalPost).then(() => {
      copied = true
      setTimeout(() => (copied = false), 2000)
    })
  }

  function formatDate(str) {
    if (!str) return ''
    try {
      return new Date(str).toLocaleDateString('en-GB', {
        day: 'numeric', month: 'short', year: 'numeric',
      })
    } catch { return str }
  }

  function stanceClass(s) {
    return 'stance-' + (s || '').toLowerCase()
  }

  function dimScore(ev, dim) {
    return ev?.dimensions?.find(d => d.name === dim)?.score ?? 0
  }

  function judgeNames(evals) {
    return Object.keys(evals || {})
  }

  function isBestIter(entry) {
    return autoHistory.length > 0 &&
      entry.avg === Math.max(...autoHistory.map(e => e.avg))
  }
</script>

<!-- ── Layout ──────────────────────────────────────────────────────────────── -->
<div class="app">

  <header>
    <div class="header-inner">
      <span class="brand-icon"></span>
      <span class="brand-name">KickstartAI</span>
      <span class="brand-sep">|</span>
      <span class="brand-sub">Thought Leadership Generator</span>
    </div>

    {#if step !== 'idle'}
      <button class="btn-ghost" on:click={reset}>Start over</button>
    {/if}
  </header>

  <main>

    {#if pipelineStage > 0}
      <nav class="pipeline-stepper">
        {#each [
          { label: 'Scan', num: 1 },
          { label: 'Interpret', num: 2 },
          { label: 'Generate', num: 3 },
          { label: 'Refine', num: 4 },
          { label: 'Done', num: 5 },
        ] as s, i}
          <div class="ps-step"
               class:ps-done={pipelineStage > s.num}
               class:ps-active={pipelineStage === s.num}>
            <div class="ps-circle">
              {#if pipelineStage > s.num}✓{:else}{s.num}{/if}
            </div>
            <span class="ps-label">{s.label}</span>
          </div>
          {#if i < 4}
            <div class="ps-line" class:ps-line-done={pipelineStage > s.num}></div>
          {/if}
        {/each}
      </nav>
    {/if}

    <!-- ── IDLE ────────────────────────────────────────────────────────────── -->
    {#if step === 'idle'}
      <div class="card center-card hero">
        <div class="hero-icon"></div>
        <h1>What's happening in AI today?</h1>
        <p class="muted">
          Find the most relevant AI news, understand it through KickstartAI's perspective,
          and turn it into a LinkedIn post — all in a few clicks.
        </p>
        <div class="hero-actions">
          <button class="btn-primary btn-lg" on:click={startScan}>
            Scan the Web
          </button>
          <button class="btn-secondary btn-lg" on:click={useCached}>
            📂 Use Last Results
          </button>
        </div>
        <p class="hero-hint">
          <strong>Scan the Web</strong> fetches fresh articles and picks the top 5 most relevant ones (~5–10 min).
          <br>
          <strong>Use Last Results</strong> skips the search and loads the most recent articles immediately.
        </p>
      </div>

    <!-- ── SCANNING ─────────────────────────────────────────────────────────── -->
    {:else if step === 'scanning'}
      <div class="card scan-card">
        <h2>Looking for today's AI news…</h2>
        <p class="muted">This takes a few minutes — we're doing the reading so you don't have to.</p>

        <div class="scan-steps">

          <!-- Step 1 – Collecting -->
          <div class="scan-step" class:active={scanPhase === 'collecting'} class:done={scanPhase === 'scoring'}>
            <div class="scan-step-num">
              {#if scanPhase === 'scoring'}
                <span class="step-check">✓</span>
              {:else}
                <span class="step-spinner-sm"></span>
              {/if}
            </div>
            <div class="scan-step-body">
              <div class="scan-step-title">Searching news sources</div>
              <div class="scan-step-desc">
                {#if scanPhase === 'collecting'}
                  Checking news sites, research papers, blogs, and RSS feeds…
                {:else}
                  Found {collectCount > 0 ? collectCount : 'all'} articles to evaluate
                {/if}
              </div>
            </div>
          </div>

          <div class="scan-step-connector"></div>

          <!-- Step 2 – Scoring -->
          <div class="scan-step" class:active={scanPhase === 'scoring'} class:waiting={scanPhase === 'collecting'}>
            <div class="scan-step-num">
              {#if scanPhase === 'collecting'}
                <span class="step-num-idle">2</span>
              {:else}
                <span class="step-spinner-sm"></span>
              {/if}
            </div>
            <div class="scan-step-body">
              <div class="scan-step-title">Scoring each article for AI relevance</div>
              <div class="scan-step-desc">
                {#if scanPhase === 'collecting'}
                  Starting soon…
                {:else}
                  An AI judge is reading and rating each article
                  {#if scoreTotal > 0}
                    <div class="inline-progress-wrap">
                      <div class="progress-track">
                        <div class="progress-fill" style="width:{Math.round(100*scoreCount/scoreTotal)}%"></div>
                      </div>
                      <span class="inline-frac">{scoreCount} / {scoreTotal} articles</span>
                    </div>
                  {/if}
                {/if}
              </div>
            </div>
          </div>

        </div>

        {#if scanWarning}
          <div class="scan-notice">
            <span class="notice-icon">ℹ️</span>
            <span>Fresh scan wasn't available — showing the most recent saved results instead.</span>
          </div>
        {/if}

        <details class="tech-details">
          <summary>Show technical log</summary>
          <div class="log-box" style="margin-top:8px">
            {#each logs as line}
              <div class="log-line">{line}</div>
            {/each}
            <div class="log-cursor">▋</div>
          </div>
        </details>
      </div>

    <!-- ── ARTICLES ──────────────────────────────────────────────────────────── -->
    {:else if step === 'articles'}
      <div class="section-head">
        <h2>📰 Top 5 AI Stories Today</h2>
        <p class="muted">Pick the story you'd like to turn into a LinkedIn post.</p>
        {#if cachedFileMeta}
          <div class="source-chip">
            📂 From saved results
            <span class="age-note">· {cachedFileMeta.age_hours}h ago</span>
          </div>
        {/if}
      </div>

      {#each articles as article, i}
        <div class="article-card">
          <div class="rank">#{i + 1}</div>
          <div class="article-body">
            <div class="score-row">
              <div class="score-bar-wrap">
                <div class="score-bar" style="width:{article.score * 10}%"></div>
              </div>
              <span class="score-label">Relevance {article.score}/10</span>
            </div>
            <h3 class="article-title">
              <a href={article.url} target="_blank" rel="noopener noreferrer">
                {article.title}
              </a>
            </h3>
            <div class="meta-row">
              <span class="badge source">{article.source}</span>
              <span class="meta-date">{formatDate(article.published_at)}</span>
              {#if article.language && article.language !== 'en'}
                <span class="badge lang">{article.language.toUpperCase()}</span>
              {/if}
            </div>
          </div>
          <button class="btn-primary" on:click={() => selectArticle(article)}>
            Select 
          </button>
        </div>
      {/each}

    <!-- ── INTERPRETING ───────────────────────────────────────────────────────── -->
    {:else if step === 'interpreting'}
      <div class="card loading-steps-card">
        <h2>Analysing the article…</h2>
        <p class="muted">Understanding what happened and how it connects to KickstartAI.</p>
        <div class="context-pill">{selectedArticle?.title}</div>
        <div class="loading-steps">
          {#each [
            { label: 'Searching the belief database', desc: 'Finding the most relevant KickstartAI beliefs using semantic search' },
            { label: "Mapping to KickstartAI's perspective", desc: 'Connecting news events to organisational stance and values' },
            { label: 'Synthesising interpretation', desc: 'Generating structured analysis with supporting arguments' },
          ] as ls, i}
            <div class="loading-step"
                 class:ls-done={interpretSubStep > i + 1}
                 class:ls-active={interpretSubStep === i + 1}
                 class:ls-waiting={interpretSubStep < i + 1}>
              <div class="ls-icon">
                {#if interpretSubStep > i + 1}
                  <span class="ls-check">✓</span>
                {:else if interpretSubStep === i + 1}
                  <span class="step-spinner-sm"></span>
                {:else}
                  <span class="ls-num">{i + 1}</span>
                {/if}
              </div>
              <div class="ls-text">
                <div class="ls-label">{ls.label}</div>
                <div class="ls-desc">{ls.desc}</div>
              </div>
            </div>
          {/each}
        </div>
      </div>

    <!-- ── INTERPRETED ────────────────────────────────────────────────────────── -->
    {:else if step === 'interpreted'}
      <div class="section-head">
        <div class="back-row">
          <button class="btn-ghost small" on:click={() => step = 'articles'}>← Back</button>
        </div>
        <h2>🧠 Here's what we found</h2>
        <p class="muted">Article: <strong>{selectedArticle?.title?.slice(0, 80)}{selectedArticle?.title?.length > 80 ? '…' : ''}</strong></p>
      </div>

      <div class="card">
        <div class="interp-stance-row">
          <span class="stance-badge {stanceClass(interpretation.interpretation['Key stance / opinion'])}">
            {interpretation.interpretation['Key stance / opinion']}
          </span>
        </div>

        <div class="interp-section">
          <h4>What happened</h4>
          <p>{interpretation.interpretation['What happened']}</p>
        </div>

        <div class="interp-section">
          <h4>Why it matters</h4>
          <p>{interpretation.interpretation['Why does it matter (globally and NL)']}</p>
        </div>

        <div class="interp-section">
          <h4>Why it matters for KickstartAI</h4>
          <p>{interpretation.interpretation['Why does it matter for KickstartAI']}</p>
        </div>

        <div class="interp-section">
          <h4>Key arguments</h4>
          <ul class="args-list">
            {#each interpretation.interpretation['Supporting arguments'] as arg}
              <li>{arg}</li>
            {/each}
          </ul>
        </div>

        {#if interpretation.matched_beliefs?.length}
          <div class="interp-section belief-section">
            <h4>Relevant KickstartAI beliefs</h4>
            {#each interpretation.matched_beliefs as b}
              <div class="belief-chip">
                <span class="belief-id">{b.belief_id}</span>
                <span class="belief-text">{b.belief_text.slice(0, 120)}{b.belief_text.length > 120 ? '…' : ''}</span>
              </div>
            {/each}
          </div>
        {/if}

        <div class="card-footer">
          <button class="btn-primary btn-lg" on:click={generatePosts}>
             Generate LinkedIn Post 
          </button>
        </div>
      </div>

    <!-- ── GENERATING ─────────────────────────────────────────────────────────── -->
    {:else if step === 'generating'}
      <div class="card loading-steps-card">
        <h2>Writing your LinkedIn post…</h2>
        <p class="muted">Crafting a post based on the article and KickstartAI's perspective.</p>
        <div class="loading-steps">
          {#each [
            { label: 'Retrieving similar posts', desc: 'Finding relevant KickstartAI examples to match tone and voice' },
            { label: "Drafting the post", desc: "Applying stance, structure, and KickstartAI's perspective" },
            { label: 'Finalising', desc: 'Polishing hashtags and call-to-action' },
          ] as ls, i}
            <div class="loading-step"
                 class:ls-done={generateSubStep > i + 1}
                 class:ls-active={generateSubStep === i + 1}
                 class:ls-waiting={generateSubStep < i + 1}>
              <div class="ls-icon">
                {#if generateSubStep > i + 1}
                  <span class="ls-check">✓</span>
                {:else if generateSubStep === i + 1}
                  <span class="step-spinner-sm"></span>
                {:else}
                  <span class="ls-num">{i + 1}</span>
                {/if}
              </div>
              <div class="ls-text">
                <div class="ls-label">{ls.label}</div>
                <div class="ls-desc">{ls.desc}</div>
              </div>
            </div>
          {/each}
        </div>
      </div>

    <!-- ── POSTS ──────────────────────────────────────────────────────────────── -->
    {:else if step === 'posts'}
      <div class="section-head">
        <div class="back-row">
          <button class="btn-ghost small" on:click={() => step = 'interpreted'}>← Back</button>
        </div>
        <h2>✍️ Choose a Post</h2>
        <p class="muted">We generated a LinkedIn post. Select it to continue.</p>
      </div>

      {#if interpretation}
        <div class="interp-banner">
          <span class="stance-badge {stanceClass(interpretation.interpretation['Key stance / opinion'])}">
            {interpretation.interpretation['Key stance / opinion']}
          </span>
          <span class="banner-summary">
            {interpretation.interpretation['What happened']?.slice(0, 160)}{interpretation.interpretation['What happened']?.length > 160 ? '…' : ''}
          </span>
        </div>
      {/if}

      {#each posts as post}
        <div class="post-card">
          <div class="post-header">
            <span class="badge post-num">Post {post.post_idx}</span>
            <span class="angle-label">{post.angle}</span>
          </div>
          <pre class="post-content">{post.content}</pre>
          <button class="btn-primary" on:click={() => selectPost(post)}>
            Select this post 
          </button>
        </div>
      {/each}

    <!-- ── POST ACTION ────────────────────────────────────────────────────────── -->
    {:else if step === 'post_action'}
      <div class="section-head">
        <div class="back-row">
          <button class="btn-ghost small" on:click={() => step = 'posts'}>← Back</button>
        </div>
        <h2>📝 What would you like to do next?</h2>
      </div>

      <div class="selected-post-preview">
        <div class="post-header">
          <span class="badge post-num">Post {selectedPost?.post_idx}</span>
          <span class="angle-label">{selectedPost?.angle}</span>
        </div>
        <pre class="post-content scrollable">{selectedPost?.content}</pre>
      </div>

      <div class="action-grid">
        <button class="action-card" on:click={startAutoRefine}>
          <div class="action-icon"></div>
          <h3>Auto-refine</h3>
          <p>Two AI judges score the post across 6 quality dimensions and rewrite it up to 3 times until it's great.</p>
        </button>

        <button class="action-card" on:click={startFeedback}>
          <div class="action-icon"></div>
          <h3>Give your feedback</h3>
          <p>Tell us what to improve in your own words — shorter, punchier, more Dutch angle, etc.</p>
        </button>

        <button class="action-card" on:click={keepPost}>
          <div class="action-icon"></div>
          <h3>Keep this post</h3>
          <p>Happy with it as-is? Copy and go.</p>
        </button>
      </div>

    <!-- ── FEEDBACK INPUT ─────────────────────────────────────────────────────── -->
    {:else if step === 'feedback_input'}
      <div class="section-head">
        <div class="back-row">
          <button class="btn-ghost small" on:click={() => step = 'post_action'}>← Back</button>
        </div>
        <h2> What would you like to change?</h2>
        <p class="muted">Describe in plain language what you'd like to improve.</p>
      </div>

      <div class="card">
        <pre class="post-content scrollable preview-bg">{selectedPost?.content}</pre>

        <textarea
          class="feedback-ta"
          bind:value={feedbackText}
          placeholder="e.g. Make it shorter and punchier, add a stronger call-to-action, focus more on the Dutch angle…"
          rows="4"
        ></textarea>

        <div class="btn-row">
          <button
            class="btn-primary"
            on:click={submitFeedback}
            disabled={!feedbackText.trim()}
          >
            Rewrite 
          </button>
        </div>
      </div>

    <!-- ── FEEDBACK REFINING ──────────────────────────────────────────────────── -->
    {:else if step === 'feedback_refining'}
      <div class="card center-card">
        <div class="spinner"></div>
        <h2>Rewriting your post…</h2>
        <p class="muted">Applying your feedback now.</p>
      </div>

    <!-- ── AUTO-REFINING ──────────────────────────────────────────────────────── -->
    {:else if step === 'auto_refining'}
      <div class="section-head">
        <h2>AI is reviewing your post…</h2>
        <p class="muted">Two independent AI judges score the post and rewrite it if needed — up to 3 rounds.</p>
      </div>

      <!-- Dimension guide -->
      <div class="dim-guide card">
        <div class="dim-guide-header">
          <span class="dim-guide-title">What the judges look for</span>
        </div>
        <div class="dim-guide-grid">
          {#each DIMENSIONS as dim}
            <div class="dim-guide-item">
              <span class="dim-guide-name">{dim.replace(/_/g, ' ')}</span>
              <span class="dim-guide-desc">{DIMENSION_GUIDE[dim]}</span>
            </div>
          {/each}
        </div>
      </div>

      <!-- Live status -->
      {#if refineStatus}
        <div class="refine-status card">
          <div class="refine-status-line">
            <span class="step-spinner-sm"></span>
            <span class="refine-status-msg">{refineStatus}</span>
          </div>
        </div>
      {/if}

      <!-- Iteration cards as they arrive -->
      {#each autoHistory as entry}
        <div class="iteration-card">
          <div class="iteration-header">
            <span class="iter-badge">Round {entry.iteration}</span>
            <span class="iter-avg">Overall score: <strong>{entry.avg} / 5</strong></span>
          </div>

          <pre class="post-content scrollable preview-bg">{entry.post}</pre>

          <div class="score-viz">
            {#each DIMENSIONS as dim}
              {@const scores = judgeNames(entry.evaluations).map(j => dimScore(entry.evaluations[j], dim))}
              <div class="sv-row">
                <span class="sv-name">{dim.replace(/_/g, ' ')}</span>
                <div class="sv-bars">
                  {#each scores as s, ji}
                    <div class="sv-judge-bar">
                      <span class="sv-judge-name">{judgeNames(entry.evaluations)[ji]}</span>
                      <div class="sv-track">
                        <div class="sv-fill"
                             class:sv-pass={s >= 4}
                             class:sv-warn={s === 3}
                             class:sv-fail={s < 3 && s > 0}
                             style="width:{s / 5 * 100}%"></div>
                      </div>
                      <span class="sv-num"
                            class:sv-pass={s >= 4}
                            class:sv-warn={s === 3}
                            class:sv-fail={s < 3 && s > 0}>{s}</span>
                    </div>
                  {/each}
                </div>
              </div>
            {/each}
            <div class="sv-overall">
              <span class="sv-name" style="font-weight:700;color:#334155">Overall</span>
              <div class="sv-bars">
                <div class="sv-judge-bar">
                  <span class="sv-judge-name"></span>
                  <div class="sv-track sv-track-lg">
                    <div class="sv-fill sv-overall-fill" style="width:{entry.avg / 5 * 100}%"></div>
                  </div>
                  <span class="sv-overall-num">{entry.avg} / 5</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      {/each}

    <!-- ── AUTO-REVIEWED ───────────────────────────────────────────────────────── -->
    {:else if step === 'auto_reviewed'}
      <div class="section-head">
        <h2>Refinement complete</h2>
        <p class="muted">
          Here are all {autoHistory.length} version{autoHistory.length !== 1 ? 's' : ''}.
          The best-scoring one is highlighted — but you can pick any version you prefer.
        </p>
      </div>

      <!-- Quick-pick best button -->
      <div class="best-pick-bar">
        <span class="best-pick-label">Best version: Round {autoHistory.reduce((b, e) => e.avg > b.avg ? e : b, autoHistory[0])?.iteration} — score {Math.max(...autoHistory.map(e => e.avg))} / 5</span>
        <button class="btn-primary" on:click={() => useAutoPost(autoFinalPost)}>
          Use best version 
        </button>
      </div>

      <!-- Dimension guide (collapsed by default here) -->
      <details class="dim-guide-detail">
        <summary>What do these scores mean?</summary>
        <div class="dim-guide-grid" style="margin-top:10px">
          {#each DIMENSIONS as dim}
            <div class="dim-guide-item">
              <span class="dim-guide-name">{dim.replace(/_/g, ' ')}</span>
              <span class="dim-guide-desc">{DIMENSION_GUIDE[dim]}</span>
            </div>
          {/each}
        </div>
      </details>

      <!-- All iteration cards -->
      {#each autoHistory as entry}
        {@const best = isBestIter(entry)}
        <div class="iteration-card" class:iter-best={best}>
          <div class="iteration-header">
            <span class="iter-badge">Round {entry.iteration}</span>
            {#if best}<span class="best-badge">Best</span>{/if}
            <span class="iter-avg">Overall score: <strong>{entry.avg} / 5</strong></span>
            <button class="btn-use-post" on:click={() => useAutoPost(entry.post)}>
              Use this version 
            </button>
          </div>

          <pre class="post-content scrollable preview-bg">{entry.post}</pre>

          <div class="score-viz">
            {#each DIMENSIONS as dim}
              {@const scores = judgeNames(entry.evaluations).map(j => dimScore(entry.evaluations[j], dim))}
              <div class="sv-row">
                <span class="sv-name">{dim.replace(/_/g, ' ')}</span>
                <div class="sv-bars">
                  {#each scores as s, ji}
                    <div class="sv-judge-bar">
                      <span class="sv-judge-name">{judgeNames(entry.evaluations)[ji]}</span>
                      <div class="sv-track">
                        <div class="sv-fill"
                             class:sv-pass={s >= 4}
                             class:sv-warn={s === 3}
                             class:sv-fail={s < 3 && s > 0}
                             style="width:{s / 5 * 100}%"></div>
                      </div>
                      <span class="sv-num"
                            class:sv-pass={s >= 4}
                            class:sv-warn={s === 3}
                            class:sv-fail={s < 3 && s > 0}>{s}</span>
                    </div>
                  {/each}
                </div>
              </div>
            {/each}
            <div class="sv-overall">
              <span class="sv-name" style="font-weight:700;color:#334155">Overall</span>
              <div class="sv-bars">
                <div class="sv-judge-bar">
                  <span class="sv-judge-name"></span>
                  <div class="sv-track sv-track-lg">
                    <div class="sv-fill sv-overall-fill" style="width:{entry.avg / 5 * 100}%"></div>
                  </div>
                  <span class="sv-overall-num">{entry.avg} / 5</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      {/each}

    <!-- ── FINAL ───────────────────────────────────────────────────────────────── -->
    {:else if step === 'final'}
      <div class="card final-card">
        <div class="final-header">
          <h2>Your LinkedIn Post</h2>
          <button class="btn-copy" class:copied on:click={copyPost}>
            {copied ? '✓ Copied!' : '📋 Copy'}
          </button>
        </div>
        <pre class="post-content final-post">{finalPost}</pre>
        <div class="btn-row">
          <button class="btn-ghost" on:click={reset}>← Start over</button>
        </div>
      </div>

    <!-- ── ERROR ───────────────────────────────────────────────────────────────── -->
    {:else if step === 'error'}
      <div class="card error-card">
        <h2>❌ Something went wrong</h2>
        <pre class="error-msg">{errorMsg}</pre>
        {#if logs.length > 0}
          <details class="error-logs">
            <summary>Show technical details ({logs.length} lines)</summary>
            <div class="log-box" style="max-height:260px;margin-top:8px">
              {#each logs as line}
                <div class="log-line">{line}</div>
              {/each}
            </div>
          </details>
        {/if}
        <button class="btn-ghost" on:click={reset}>← Start over</button>
      </div>
    {/if}

  </main>
</div>

<!-- ── Styles ──────────────────────────────────────────────────────────────── -->
<style>
  :global(*) { box-sizing: border-box; margin: 0; padding: 0; }
  :global(body) {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #f0f4f8;
    color: #1e293b;
    min-height: 100vh;
  }
@keyframes gradientShift {
  0% {
    background-position: 0% 50%;
  }
  50% {
    background-position: 100% 50%;
  }
  100% {
    background-position: 0% 50%;
  }
  
}
  /* ── App shell ─────────────────────────────────────────────────────────── */
  .app { display: flex; flex-direction: column; min-height: 100vh; }

  header {
  height: 52px;
  padding: 0 24px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  position: sticky;
  top: 0;
  z-index: 10;

  /* animated gradient */
  background: linear-gradient(
    270deg,
    #3b82f6,
    #a855f7,
    #facc15,
    #3b82f6
  );

  background-size: 600% 600%;
  animation: gradientShift 15s ease infinite;

  border-bottom: none;
  box-shadow: 0 2px 10px rgba(0,0,0,0.15);
  }
  
  .header-inner { display: flex; align-items: center; gap: 10px; }
  .brand-icon { font-size: 20px; }
  .brand-name { font-weight: 700; font-size: 17px; color: #ffffff; letter-spacing: -.3px; }
  .brand-sep  { color: #cbd5e1; }
  .brand-sub  { color: white; font-size: 13px; }

  main { max-width: 780px; margin: 0 auto; padding: 28px 16px 60px; width: 100%; }

  /* ── Cards ─────────────────────────────────────────────────────────────── */
  .card {
    background: #fff;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 16px;
    box-shadow: 0 1px 4px rgba(0,0,0,.07);
  }
  .center-card { text-align: center; padding: 52px 24px; }
  .card-footer { margin-top: 20px; }

  h1 { font-size: 26px; font-weight: 700; margin-bottom: 10px; }
  h2 { font-size: 20px; font-weight: 700; margin-bottom: 12px; }
  h3 { font-size: 15px; font-weight: 600; margin-bottom: 6px; }
  h4 { font-size: 13px; font-weight: 600; color: #475569; margin-bottom: 6px; text-transform: uppercase; letter-spacing: .4px; }
  p  { line-height: 1.6; color: #334155; }
  .muted { color: #64748b; font-size: 14px; margin-bottom: 24px; line-height: 1.5; }

  /* ── Buttons ───────────────────────────────────────────────────────────── */
  .btn-primary {
    background: #0077b5; color: #fff; border: none;
    padding: 10px 22px; border-radius: 8px; cursor: pointer;
    font-size: 14px; font-weight: 600;
    transition: background .15s, transform .1s;
    white-space: nowrap;
  }
  .btn-primary:hover:not(:disabled) { background: #005f94; transform: translateY(-1px); }
  .btn-primary:disabled { background: #94a3b8; cursor: not-allowed; }
  .btn-lg { padding: 14px 32px; font-size: 16px; }

  .btn-ghost {
    background: transparent; color: #475569; border: 1px solid #e2e8f0;
    padding: 8px 18px; border-radius: 8px; cursor: pointer;
    font-size: 14px; font-weight: 500;
    transition: background .12s;
  }
  .btn-ghost:hover { background: #f1f5f9; }
  .btn-ghost.small { padding: 5px 12px; font-size: 13px; }

  .btn-copy {
    background: #f1f5f9; color: #0077b5;
    border: 1px solid #0077b5;
    padding: 8px 18px; border-radius: 8px; cursor: pointer;
    font-size: 14px; font-weight: 600;
    transition: all .15s;
  }
  .btn-copy.copied { background: #dcfce7; color: #166534; border-color: #86efac; }
  .btn-row { display: flex; gap: 10px; justify-content: flex-end; margin-top: 16px; }

  .btn-use-post {
    margin-left: auto;
    background: #f0f9ff; color: #0077b5;
    border: 1px solid #bae6fd;
    padding: 6px 14px; border-radius: 7px; cursor: pointer;
    font-size: 13px; font-weight: 600;
    transition: all .12s;
    white-space: nowrap;
  }
  .btn-use-post:hover { background: #0077b5; color: #fff; border-color: #0077b5; }

  /* ── Hero ──────────────────────────────────────────────────────────────── */
  .hero-icon { font-size: 52px; margin-bottom: 16px; }
  .hero-actions {
    display: flex; gap: 14px; justify-content: center;
    flex-wrap: wrap; margin-bottom: 16px;
  }
  .btn-secondary {
    background: #fff; color: #0077b5;
    border: 2px solid #0077b5;
    padding: 10px 22px; border-radius: 8px; cursor: pointer;
    font-size: 14px; font-weight: 600;
    transition: background .15s, transform .1s;
    white-space: nowrap;
  }
  .btn-secondary:hover { background: #f0f9ff; transform: translateY(-1px); }
  .btn-secondary.btn-lg { padding: 14px 32px; font-size: 16px; }
  .hero-hint {
    font-size: 13px; color: #94a3b8; line-height: 1.6;
    max-width: 420px; margin: 0 auto;
  }

  /* ── Spinner ───────────────────────────────────────────────────────────── */
  .spinner {
    width: 40px; height: 40px;
    border: 3px solid #e2e8f0; border-top-color: #0077b5;
    border-radius: 50%;
    animation: spin .8s linear infinite;
    margin: 0 auto 20px;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* ── Small inline spinner ──────────────────────────────────────────────── */
  .step-spinner-sm {
    display: inline-block;
    width: 18px; height: 18px;
    border: 2px solid #e2e8f0; border-top-color: #0077b5;
    border-radius: 50%;
    animation: spin .8s linear infinite;
    flex-shrink: 0;
  }

  /* ── Scan card ─────────────────────────────────────────────────────────── */
  .scan-card { padding-bottom: 20px; }

  .scan-steps {
    display: flex;
    flex-direction: column;
    gap: 0;
    margin: 24px 0 20px;
  }

  .scan-step {
    display: flex;
    align-items: flex-start;
    gap: 16px;
    padding: 18px 20px;
    border-radius: 10px;
    border: 1px solid #e2e8f0;
    background: #fafafa;
    transition: all .25s;
  }
  .scan-step.active {
    border-color: #0077b5;
    background: #f0f9ff;
  }
  .scan-step.done {
    border-color: #86efac;
    background: #f0fdf4;
  }
  .scan-step.waiting {
    opacity: 0.5;
  }

  .scan-step-connector {
    width: 2px; height: 14px;
    background: #e2e8f0;
    margin-left: 28px;
  }

  .scan-step-num {
    width: 36px; height: 36px;
    border-radius: 50%;
    background: #e2e8f0;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 14px; color: #94a3b8;
    flex-shrink: 0;
  }
  .scan-step.active .scan-step-num {
    background: #0077b5;
  }
  .scan-step.done .scan-step-num {
    background: #22c55e;
    color: #fff;
  }

  .step-check { color: #fff; font-size: 16px; }
  .step-num-idle { color: #94a3b8; font-size: 14px; font-weight: 700; }

  .scan-step-body { flex: 1; }
  .scan-step-title {
    font-weight: 600; font-size: 15px; color: #1e293b;
    margin-bottom: 4px;
  }
  .scan-step.waiting .scan-step-title { color: #94a3b8; }
  .scan-step-desc {
    font-size: 13px; color: #64748b; line-height: 1.5;
  }

  .inline-progress-wrap {
    display: flex; align-items: center; gap: 10px;
    margin-top: 10px;
  }
  .progress-track {
    flex: 1; height: 6px;
    background: #e2e8f0; border-radius: 4px;
    overflow: hidden;
  }
  .progress-fill {
    height: 100%; background: linear-gradient(90deg, #0077b5, #38bdf8);
    border-radius: 4px; transition: width 0.4s ease;
  }
  .inline-frac { font-size: 12px; font-weight: 600; color: #475569; white-space: nowrap; }

  /* ── Scan notice (fallback warning) ───────────────────────────────────── */
  .scan-notice {
    display: flex; align-items: flex-start; gap: 10px;
    background: #fffbeb; border: 1px solid #fde68a;
    border-radius: 8px; padding: 12px 14px;
    font-size: 13px; color: #92400e;
    margin-bottom: 14px;
  }
  .notice-icon { flex-shrink: 0; }

  /* ── Technical details toggle ──────────────────────────────────────────── */
  .tech-details { margin-top: 8px; }
  .tech-details summary,
  .dim-guide-detail summary {
    cursor: pointer; font-size: 13px; color: #94a3b8;
    user-select: none; list-style: none;
  }
  .tech-details summary:hover,
  .dim-guide-detail summary:hover { color: #475569; }

  /* ── Log box (kept for tech details & error) ───────────────────────────── */
  .log-box {
    background: #0f172a;
    border-radius: 8px; padding: 14px;
    max-height: 320px; overflow-y: auto;
    font-family: 'Menlo', 'Courier New', monospace;
    font-size: 12px; color: #94a3b8;
  }
  .log-line { margin-bottom: 3px; }
  .log-cursor { color: #38bdf8; animation: blink 1s step-end infinite; }
  @keyframes blink { 50% { opacity: 0; } }

  /* ── Section header ────────────────────────────────────────────────────── */
  .section-head { margin-bottom: 16px; }
  .back-row { margin-bottom: 8px; }

  /* ── Context pill ──────────────────────────────────────────────────────── */
  .context-pill {
    display: inline-block;
    background: #f1f5f9; color: #475569;
    border-radius: 20px; padding: 6px 16px;
    font-size: 13px; margin-top: 16px;
    max-width: 100%; word-break: break-word;
  }

  /* ── Articles ──────────────────────────────────────────────────────────── */
  .article-card {
    background: #fff;
    border-radius: 12px; padding: 18px 20px;
    margin-bottom: 12px;
    box-shadow: 0 1px 4px rgba(0,0,0,.07);
    display: flex; align-items: flex-start; gap: 16px;
  }
  .rank {
    font-size: 22px; font-weight: 800; color: #cbd5e1;
    min-width: 36px; padding-top: 2px;
  }
  .article-body { flex: 1; min-width: 0; }
  .score-row { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }
  .score-bar-wrap {
    flex: 1; height: 6px;
    background: #e2e8f0; border-radius: 3px; overflow: hidden;
  }
  .score-bar { height: 100%; background: #0077b5; border-radius: 3px; }
  .score-label { font-size: 12px; font-weight: 700; color: #475569; white-space: nowrap; }
  .article-title { font-size: 15px; line-height: 1.4; margin-bottom: 8px; }
  .article-title a { color: #1e293b; text-decoration: none; }
  .article-title a:hover { color: #0077b5; text-decoration: underline; }
  .meta-row { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
  .meta-date { color: #94a3b8; font-size: 12px; }

  /* ── Badges ────────────────────────────────────────────────────────────── */
  .badge {
    display: inline-block; padding: 2px 8px;
    border-radius: 4px; font-size: 11px; font-weight: 700;
    letter-spacing: .3px;
  }
  .badge.source { background: #f1f5f9; color: #475569; }
  .badge.lang   { background: #fef3c7; color: #92400e; }
  .badge.post-num { background: #0077b5; color: #fff; padding: 3px 10px; }

  /* ── Stance ────────────────────────────────────────────────────────────── */
  .stance-badge {
    display: inline-block; padding: 4px 12px;
    border-radius: 20px; font-size: 12px; font-weight: 700;
    text-transform: uppercase; letter-spacing: .6px;
  }
  .stance-supportive { background: #dcfce7; color: #166534; }
  .stance-critical   { background: #fee2e2; color: #991b1b; }
  .stance-cautious   { background: #fef3c7; color: #92400e; }
  .stance-neutral    { background: #f1f5f9; color: #475569; }

  /* ── Interpretation ────────────────────────────────────────────────────── */
  .interp-stance-row { margin-bottom: 20px; }
  .interp-section {
    border-top: 1px solid #f1f5f9;
    padding: 14px 0;
  }
  .interp-section p, .interp-section li { font-size: 14px; line-height: 1.65; color: #334155; }
  .args-list { padding-left: 20px; }
  .args-list li { margin-bottom: 6px; }
  .belief-chip {
    display: flex; gap: 8px; align-items: flex-start;
    background: #f8fafc; border-radius: 6px;
    padding: 8px 12px; margin-bottom: 6px;
    font-size: 13px;
  }
  .belief-id   { color: #0077b5; font-weight: 700; white-space: nowrap; }
  .belief-text { color: #475569; }

  /* ── Interpretation banner (above posts) ───────────────────────────────── */
  .interp-banner {
    display: flex; align-items: flex-start; gap: 12px;
    background: #f8fafc; border-left: 4px solid #0077b5;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px; margin-bottom: 16px;
  }
  .banner-summary { font-size: 13px; color: #475569; line-height: 1.5; flex: 1; }

  /* ── Posts ─────────────────────────────────────────────────────────────── */
  .post-card {
    background: #fff; border-radius: 12px;
    padding: 22px; margin-bottom: 14px;
    box-shadow: 0 1px 4px rgba(0,0,0,.07);
  }
  .post-header { display: flex; align-items: center; gap: 10px; margin-bottom: 14px; }
  .angle-label { color: #64748b; font-size: 13px; font-style: italic; }
  .post-content {
    white-space: pre-wrap; font-family: inherit;
    font-size: 14px; line-height: 1.75; color: #1e293b;
    margin-bottom: 16px;
  }
  .post-content.scrollable { max-height: 220px; overflow-y: auto; }
  .post-content.preview-bg {
    background: #f8fafc; border-radius: 8px; padding: 14px;
  }
  .post-content.final-post {
    background: #f0f9ff; border-radius: 8px;
    padding: 20px; margin-bottom: 0;
  }

  /* ── Selected post preview ─────────────────────────────────────────────── */
  .selected-post-preview {
    background: #fff; border-radius: 12px;
    padding: 20px; margin-bottom: 16px;
    box-shadow: 0 1px 4px rgba(0,0,0,.07);
  }

  /* ── Action grid ───────────────────────────────────────────────────────── */
  .action-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
  }
  .action-card {
    background: #fff; border: 2px solid #e2e8f0;
    border-radius: 12px; padding: 22px 16px;
    cursor: pointer; transition: all .15s;
    text-align: center;
    display: flex; flex-direction: column; align-items: center;
  }
  .action-card:hover {
    border-color: #0077b5;
    transform: translateY(-2px);
    box-shadow: 0 4px 14px rgba(0,119,181,.14);
  }
  .action-icon { font-size: 34px; margin-bottom: 12px; }
  .action-card h3 { font-size: 15px; margin-bottom: 8px; }
  .action-card p  { font-size: 13px; color: #64748b; line-height: 1.4; }

  /* ── Feedback textarea ─────────────────────────────────────────────────── */
  .feedback-ta {
    width: 100%; border: 1px solid #e2e8f0; border-radius: 8px;
    padding: 12px; font-family: inherit; font-size: 14px;
    line-height: 1.5; resize: vertical; margin-top: 14px;
    color: #1e293b;
  }
  .feedback-ta:focus { outline: none; border-color: #0077b5; }

  /* ── Dimension guide ───────────────────────────────────────────────────── */
  .dim-guide {
    padding: 18px 20px;
    margin-bottom: 16px;
  }
  .dim-guide-header { margin-bottom: 12px; }
  .dim-guide-title {
    font-weight: 700; font-size: 14px; color: #334155;
  }
  .dim-guide-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
  }
  .dim-guide-item {
    background: #f8fafc; border-radius: 8px;
    padding: 10px 12px;
  }
  .dim-guide-name {
    display: block;
    font-size: 12px; font-weight: 700;
    color: #0077b5; text-transform: capitalize;
    margin-bottom: 3px;
  }
  .dim-guide-desc {
    font-size: 12px; color: #64748b; line-height: 1.4;
  }

  .dim-guide-detail {
    background: #fff; border-radius: 10px;
    padding: 14px 18px; margin-bottom: 14px;
    box-shadow: 0 1px 4px rgba(0,0,0,.07);
  }

  /* ── Refine live status ────────────────────────────────────────────────── */
  .refine-status {
    padding: 14px 18px; margin-bottom: 14px;
  }
  .refine-status-line {
    display: flex; align-items: center; gap: 12px;
  }
  .refine-status-msg {
    font-size: 14px; color: #334155;
  }

  /* ── Auto-refine iterations ────────────────────────────────────────────── */
  .iteration-card {
    background: #fff; border-radius: 10px;
    padding: 18px; margin-bottom: 14px;
    box-shadow: 0 1px 4px rgba(0,0,0,.07);
    border: 2px solid transparent;
    transition: border-color .2s;
  }
  .iteration-card.iter-best {
    border-color: #22c55e;
  }
  .iteration-header {
    display: flex; align-items: center; gap: 10px; margin-bottom: 14px;
    flex-wrap: wrap;
  }
  .iter-badge {
    background: #e0f2fe; color: #0369a1;
    padding: 3px 10px; border-radius: 20px;
    font-size: 12px; font-weight: 700;
  }
  .best-badge {
    background: #dcfce7; color: #166534;
    padding: 3px 10px; border-radius: 20px;
    font-size: 12px; font-weight: 700;
  }
  .iter-avg { font-size: 14px; color: #64748b; }
  .iter-avg strong { color: #1e293b; }

  /* ── Best-pick bar ─────────────────────────────────────────────────────── */
  .best-pick-bar {
    display: flex; align-items: center; justify-content: space-between;
    gap: 14px; flex-wrap: wrap;
    background: #f0fdf4; border: 1px solid #86efac;
    border-radius: 10px; padding: 14px 18px;
    margin-bottom: 16px;
  }
  .best-pick-label {
    font-size: 14px; font-weight: 600; color: #166534;
  }

  /* ── Score table ───────────────────────────────────────────────────────── */
  .score-table {
    width: 100%; border-collapse: collapse;
    font-size: 13px; margin-top: 14px;
  }
  .score-table th {
    background: #f8fafc; padding: 8px 10px;
    text-align: left; font-weight: 600;
    color: #475569; border-bottom: 2px solid #e2e8f0;
  }
  .score-table td { padding: 6px 10px; border-top: 1px solid #f1f5f9; }
  .dim-name { text-transform: capitalize; color: #64748b; }
  .row-avg  { color: #64748b; font-style: italic; }
  .score-table td.pass { color: #16a34a; font-weight: 700; }
  .score-table td.fail { color: #dc2626; }
  .avg-row td { border-top: 2px solid #e2e8f0; font-weight: 700; color: #475569; }

  /* ── Final card ────────────────────────────────────────────────────────── */
  .final-header {
    display: flex; justify-content: space-between;
    align-items: center; margin-bottom: 18px;
  }

  /* ── Source chip ───────────────────────────────────────────────────────── */
  .source-chip {
    display: inline-flex; align-items: center; gap: 4px;
    background: #f1f5f9; color: #475569;
    border-radius: 6px; padding: 4px 10px;
    font-size: 12px;
    margin-top: 4px;
  }
  .age-note { color: #94a3b8; }

  /* ── Error card ────────────────────────────────────────────────────────── */
  .error-card { border-left: 4px solid #ef4444; }
  .error-msg {
    white-space: pre-wrap; word-break: break-all;
    background: #fef2f2; color: #b91c1c;
    border-radius: 6px; padding: 12px;
    font-size: 12px; font-family: monospace;
    margin: 12px 0;
  }
  .error-logs { margin: 8px 0 14px; }
  .error-logs summary {
    cursor: pointer; font-size: 13px; color: #64748b;
    user-select: none;
  }
  .error-logs summary:hover { color: #1e293b; }

  /* ── Pipeline stepper ──────────────────────────────────────────────────── */
  .pipeline-stepper {
    display: flex; align-items: flex-start;
    margin-bottom: 24px; padding: 0 4px;
  }
  .ps-step {
    display: flex; flex-direction: column; align-items: center;
    gap: 5px; flex-shrink: 0;
  }
  .ps-circle {
    width: 30px; height: 30px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 12px; font-weight: 700;
    background: #e2e8f0; color: #94a3b8;
    transition: all .2s;
  }
  .ps-step.ps-done .ps-circle   { background: #22c55e; color: #fff; }
  .ps-step.ps-active .ps-circle {
    background: #0077b5; color: #fff;
    box-shadow: 0 0 0 3px rgba(0,119,181,.18);
  }
  .ps-label { font-size: 10px; font-weight: 600; color: #94a3b8; white-space: nowrap; }
  .ps-step.ps-done .ps-label   { color: #22c55e; }
  .ps-step.ps-active .ps-label { color: #0077b5; }
  .ps-line {
    flex: 1; height: 2px; background: #e2e8f0;
    margin: 14px 4px 0; transition: background .2s;
  }
  .ps-line.ps-line-done { background: #22c55e; }

  /* ── Loading steps ──────────────────────────────────────────────────────── */
  .loading-steps-card { padding: 28px 24px; }
  .loading-steps { display: flex; flex-direction: column; margin-top: 24px; }
  .loading-step {
    display: flex; gap: 14px; align-items: flex-start;
    padding: 14px 0; border-bottom: 1px solid #f1f5f9;
  }
  .loading-step:last-child { border-bottom: none; }
  .loading-step.ls-waiting { opacity: 0.45; }
  .ls-icon {
    width: 32px; height: 32px; border-radius: 50%;
    background: #e2e8f0;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0; transition: background .2s;
  }
  .loading-step.ls-active .ls-icon { background: #0077b5; }
  .loading-step.ls-done   .ls-icon { background: #22c55e; }
  .ls-check { color: #fff; font-size: 14px; font-weight: 700; }
  .ls-num   { color: #94a3b8; font-size: 13px; font-weight: 700; }
  .ls-text  { flex: 1; padding-top: 5px; }
  .ls-label { font-size: 14px; font-weight: 600; color: #1e293b; margin-bottom: 3px; }
  .ls-desc  { font-size: 12px; color: #94a3b8; line-height: 1.4; }
  .loading-step.ls-active .ls-label { color: #0077b5; }
  .loading-step.ls-done   .ls-label { color: #16a34a; }

  /* ── Score visualisation ────────────────────────────────────────────────── */
  .score-viz { margin-top: 16px; display: flex; flex-direction: column; gap: 10px; }
  .sv-row { display: flex; align-items: center; gap: 10px; }
  .sv-name {
    width: 140px; flex-shrink: 0;
    font-size: 12px; font-weight: 600; color: #64748b;
    text-transform: capitalize;
  }
  .sv-bars  { flex: 1; display: flex; flex-direction: column; gap: 4px; }
  .sv-judge-bar { display: flex; align-items: center; gap: 8px; }
  .sv-judge-name { width: 48px; font-size: 11px; color: #94a3b8; flex-shrink: 0; }
  .sv-track {
    flex: 1; height: 8px;
    background: #f1f5f9; border-radius: 4px; overflow: hidden;
  }
  .sv-track-lg { height: 10px; }
  .sv-fill {
    height: 100%; border-radius: 4px;
    background: #cbd5e1; transition: width .5s ease;
  }
  .sv-fill.sv-pass { background: #22c55e; }
  .sv-fill.sv-warn { background: #f59e0b; }
  .sv-fill.sv-fail { background: #ef4444; }
  .sv-overall-fill { background: linear-gradient(90deg, #0077b5, #22c55e); }
  .sv-num {
    width: 22px; font-size: 12px; font-weight: 700;
    text-align: right; flex-shrink: 0; color: #94a3b8;
  }
  .sv-num.sv-pass { color: #16a34a; }
  .sv-num.sv-warn { color: #b45309; }
  .sv-num.sv-fail { color: #dc2626; }
  .sv-overall {
    display: flex; align-items: center; gap: 10px;
    padding-top: 10px; border-top: 2px solid #e2e8f0; margin-top: 4px;
  }
  .sv-overall-num { font-size: 13px; font-weight: 700; color: #1e293b; white-space: nowrap; flex-shrink: 0; }

  /* ── Responsive ────────────────────────────────────────────────────────── */
  @media (max-width: 600px) {
    .action-grid { grid-template-columns: 1fr; }
    .article-card { flex-direction: column; }
    .btn-lg { width: 100%; }
    .hero-actions { flex-direction: column; align-items: stretch; }
    .dim-guide-grid { grid-template-columns: 1fr; }
    .best-pick-bar { flex-direction: column; align-items: flex-start; }
    .iteration-header { flex-direction: column; align-items: flex-start; }
    .btn-use-post { margin-left: 0; }
  }
</style>
