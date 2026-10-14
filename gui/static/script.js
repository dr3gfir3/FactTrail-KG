// Global network states
let fullKGNetwork = null;
let fullKGData = null; // Cache of nodes and edges
let fullKGMode = 'communities';
let selectedKGCommunity = null;
let activeBenchmarkInterval = null;
let wikiReaderPages = [];
let wikiReaderNetwork = null;

document.addEventListener('DOMContentLoaded', () => {
  setupNavigation();
  setupFactChecker();
  setupKGExplorer();
  setupWikiExplorer();
  setupBenchmarkCenter();
  setupNERPlayground();
  
  // Initial data loads
  loadWikiPages();
  loadFullKG();
  loadBenchmarkRunsList();
  checkActiveBenchmark();
});

/* 1. Sidebar Navigation */
function setupNavigation() {
  document.querySelectorAll('.menu-item').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.menu-item').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.content-view').forEach(v => v.classList.remove('active'));
      
      btn.classList.add('active');
      const targetView = 'view-' + btn.dataset.view;
      const viewEl = document.getElementById(targetView);
      if (viewEl) {
        viewEl.classList.add('active');
      }
      
      // Auto refresh actions on tab click
      if (btn.dataset.view === 'kg-explorer') {
        loadFullKG();
        loadWikiPages();
      } else if (btn.dataset.view === 'wiki-explorer') {
        loadWikiReaderPages();
      } else if (btn.dataset.view === 'benchmark-center') {
        loadBenchmarkRunsList();
      } else if (btn.dataset.view === 'system-info') {
        loadModelStatus();
      }
    });
  });
}

async function loadModelStatus() {
  try {
    const res = await fetch('/api/model-status');
    const data = await res.json();
    
    const statusMap = {
      'spacy': 'status-spacy',
      'sbert': 'status-sbert',
      'deberta': 'status-deberta'
    };
    
    Object.entries(statusMap).forEach(([key, elemId]) => {
      const elem = document.getElementById(elemId);
      if (elem && data[key]) {
        const model = data[key];
        elem.textContent = model.status;
        elem.className = 'badge ' + (model.loaded ? 'badge-success' : 'badge-danger');
      }
    });
  } catch (err) {
    console.error('Failed to load model status:', err);
  }
  
  // Load KG path from API
  try {
    const kgRes = await fetch('/api/kg-data');
    if (kgRes.ok) {
      const kgData = await kgRes.json();
      const kgPathElem = document.getElementById('info-kg-path');
      if (kgPathElem && kgData.kg_path) {
        kgPathElem.textContent = kgData.kg_path;
      }
    }
  } catch (err) {
    console.error('Failed to load KG path:', err);
  }
}

/* 2. Fact-Checking pipeline */
function setupFactChecker() {
  const form = document.getElementById('verify-form');
  const fileInput = document.getElementById('file');
  const fileName = document.getElementById('file-name');
  const uploadArea = document.getElementById('upload-area');

  fileInput.addEventListener('change', () => {
    fileName.textContent = fileInput.files.length ? fileInput.files[0].name : 'Click or drag file to upload';
  });

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const btn = document.getElementById('verify-btn');
    const spinner = btn.querySelector('.spinner');
    const btnText = btn.querySelector('.btn-text');
    
    btn.disabled = true;
    spinner.classList.remove('hidden');
    btnText.textContent = 'Verifying...';
    document.getElementById('results').classList.add('hidden');
    
    // Clear Claim graph placeholder
    const graphContainer = document.getElementById('claim-kg-visualization');
    graphContainer.innerHTML = '<div class="graph-placeholder"><span class="spinner"></span><p style="margin-top:10px">Querying Knowledge Graph & Wikipedia...</p></div>';

    const fd = new FormData(form);
    try {
      const res = await fetch('/verify', { method: 'POST', body: fd });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.error || `HTTP ${res.status}`);
      }
      const data = await res.json();
      
      // Display Results
      showVerifyResults(data);
      
      // Load current full graph and render claim-specific subgraph
      if (!fullKGData) {
        const kgRes = await fetch('/api/kg-data');
        if (kgRes.ok) {
          fullKGData = await kgRes.json();
        }
      }
      const graphEntities = getPrimaryEntityTexts(data);
      renderClaimKG(graphEntities.length ? graphEntities : (data.entities || []), fullKGData);
    } catch (err) {
      alert('Verification Error: ' + err.message);
      graphContainer.innerHTML = '<div class="graph-placeholder"><p style="color:var(--red)">Error: ' + err.message + '</p></div>';
    } finally {
      btn.disabled = false;
      spinner.classList.add('hidden');
      btnText.textContent = 'Verify Claim';
    }
  });

  setupSubtabs();
}

function setupSubtabs() {
  document.querySelectorAll('.tab-sub-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const parent = btn.closest('.results-card');
      parent.querySelectorAll('.tab-sub-btn').forEach(b => b.classList.remove('active'));
      parent.querySelectorAll('.subtab-pane').forEach(p => p.classList.add('hidden'));
      
      btn.classList.add('active');
      const paneId = 'subtab-' + btn.dataset.subtab;
      document.getElementById(paneId).classList.remove('hidden');
    });
  });
}

function showVerifyResults(data) {
  document.getElementById('results').classList.remove('hidden');
  
  // Verdict banner styling
  const banner = document.getElementById('verdict-banner');
  const cls = data.verdict.replace(/\s+/g, '_');
  banner.className = 'verdict-banner ' + cls;
  
  document.getElementById('verdict-value').textContent = data.verdict;
  document.getElementById('confidence-value').textContent = `${(data.confidence * 100).toFixed(1)}%`;
  
  // Pipeline metadata
  document.getElementById('pipeline-meta').style.display = 'flex';
  document.getElementById('meta-source').textContent = data.source || 'unknown';
  document.getElementById('meta-wiki-pages').textContent = data.wiki_pages_added ?? 0;
  document.getElementById('meta-wiki-triples').textContent = data.wiki_triples ?? 0;
  document.getElementById('meta-nli-label').textContent = data.nli_label || 'unknown';
  document.getElementById('meta-cache').textContent = data.cache_hit ? 'hit' : 'miss';
  document.getElementById('meta-relation-mode').textContent =
    data.relation_extraction_label || data.relation_extraction_method || 'config';
  
  // Probability distribution bar
  const probBar = document.getElementById('prob-bar');
  probBar.innerHTML = '';
  const probs = data.probs || {};
  const labels = ['SUPPORTED', 'REFUTED', 'NOT ENOUGH INFO'];
  
  labels.forEach(label => {
    const pct = (probs[label] || 0) * 100;
    if (pct > 0) {
      const seg = document.createElement('div');
      seg.className = `prob-segment ${label.replace(/\s+/g, '_')}`;
      seg.style.width = pct + '%';
      seg.textContent = pct >= 8 ? `${pct.toFixed(0)}%` : '';
      seg.title = `${label}: ${pct.toFixed(1)}%`;
      probBar.appendChild(seg);
    }
  });
  
  renderNliPremise(data);
  renderEvidenceTrace(data);
  
  renderEntityDiagnostics(data);
  
  // Wiki pages references fetched
  const refList = document.getElementById('wiki-ref-list');
  refList.innerHTML = '';
  if (data.fetched_pages && data.fetched_pages.length) {
    data.fetched_pages.forEach(p => {
      const li = document.createElement('li');
      li.textContent = p;
      refList.appendChild(li);
    });
    document.getElementById('wiki-ref-container').classList.remove('hidden');
  } else {
    document.getElementById('wiki-ref-container').classList.add('hidden');
  }
  
  // Timing details
  const timingBars = document.getElementById('timing-bars');
  timingBars.innerHTML = '';
  const stages = data.stages || {};
  const totalTime = stages.total || 1.0;
  
  const stageLabels = {
    nlp_load: 'spaCy Load',
    entity_extraction: 'Entity Extraction',
    wiki_kg_query: 'KG Search',
    sbert_load: 'SBERT Embed Load',
    evidence_ranking: 'Evidence Ranking',
    nli_load: 'DeBERTa NLI Load',
    nli_inference: 'NLI Inference',
    wiki_reload: 'Wiki Fetch Ingestion',
    total: 'Total Duration'
  };
  
  Object.entries(stages).forEach(([key, val]) => {
    if (key === 'total') return;
    const row = document.createElement('div');
    row.className = 'timing-row';
    const pct = (val / totalTime) * 100;
    
    row.innerHTML = `
      <span class="timing-lbl">${stageLabels[key] || key}</span>
      <div class="timing-bar-bg">
        <div class="timing-bar-fill" style="width: ${pct}%"></div>
      </div>
      <span class="timing-val">${val.toFixed(2)}s</span>
    `;
    timingBars.appendChild(row);
  });
  
  // Add total row
  const tRow = document.createElement('div');
  tRow.className = 'timing-row';
  tRow.style.fontWeight = '600';
  tRow.innerHTML = `
    <span class="timing-lbl" style="color:#fff">Total Duration</span>
    <div class="timing-bar-bg" style="background:rgba(255,255,255,0.06)"></div>
    <span class="timing-val" style="color:#fff">${totalTime.toFixed(2)}s</span>
  `;
  timingBars.appendChild(tRow);
  
  // Click overview tab
  document.querySelector('.tab-sub-btn[data-subtab="premise"]').click();
}

function getPrimaryEntityTexts(data) {
  return (data.entity_diagnostics || [])
    .filter(ent => ent.role === 'primary')
    .map(ent => ent.text);
}

function renderEntityDiagnostics(data) {
  const entList = document.getElementById('entities-list');
  entList.innerHTML = '';

  const diagnostics = data.entity_diagnostics || (data.entities || []).map(text => ({ text, role: 'primary' }));
  if (!diagnostics.length) {
    entList.innerHTML = '<span class="text-muted" style="font-size:0.85rem">No entities extracted.</span>';
    return;
  }

  diagnostics.forEach(ent => {
    const span = document.createElement('span');
    const role = ent.role || 'primary';
    span.className = `entity-tag entity-role-${role}`;
    span.title = [
      ent.label ? `label: ${ent.label}` : '',
      ent.discard_reason ? `reason: ${ent.discard_reason}` : ''
    ].filter(Boolean).join(' | ');
    span.innerHTML = `
      <span>${escapeHtml(ent.text || '')}</span>
      <small>${escapeHtml(role)}</small>
    `;
    entList.appendChild(span);
  });
}

function renderEvidenceTrace(data) {
  const list = document.getElementById('evidence-trace-list');
  if (!list) return;

  const trace = data.evidence_trace || [];
  list.innerHTML = '';

  if (!trace.length) {
    list.innerHTML = '<div class="empty-trace">No ranked evidence was available for this claim.</div>';
    return;
  }

  trace.forEach(item => {
    const row = document.createElement('div');
    row.className = 'trace-item';

    const score = item.cross_encoder_score ?? item.score;
    const scoreText = typeof score === 'number' ? score.toFixed(3) : 'n/a';
    const nli = item.nli_verdict
      ? `${item.nli_verdict} ${(Number(item.nli_confidence || 0) * 100).toFixed(1)}%`
      : 'NLI pending/legacy';

    row.innerHTML = `
      <div class="trace-rank">#${item.rank}</div>
      <div class="trace-content">
        <div class="trace-meta">
          <span>${escapeHtml(item.source_type || 'unknown')}</span>
          <span>score ${escapeHtml(scoreText)}</span>
          <span>${escapeHtml(nli)}</span>
        </div>
        <div class="trace-text">${escapeHtml(item.text || '')}</div>
        <div class="trace-source">${escapeHtml(item.source || item.title || 'No source label')}</div>
      </div>
    `;
    list.appendChild(row);
  });
}

function renderClaimKG(claimEntities, allKGData) {
  const container = document.getElementById('claim-kg-visualization');
  container.innerHTML = '';
  
  if (!allKGData || !allKGData.nodes || !allKGData.nodes.length) {
    container.innerHTML = '<div class="graph-placeholder"><p>No KG triples available.</p></div>';
    return;
  }
  
  const entitySet = new Set(claimEntities.map(e => e.toLowerCase()));
  
  // Filter edges connected to any claim entities
  const filteredEdges = allKGData.edges.filter(edge => {
    return entitySet.has(edge.from.toLowerCase()) || entitySet.has(edge.to.toLowerCase());
  });
  
  // Build node set
  const nodeIds = new Set();
  filteredEdges.forEach(edge => {
    nodeIds.add(edge.from);
    nodeIds.add(edge.to);
  });
  
  claimEntities.forEach(ent => {
    // Find node matching entity
    const matched = allKGData.nodes.find(n => n.id.toLowerCase() === ent.toLowerCase());
    if (matched) {
      nodeIds.add(matched.id);
    }
  });
  
  const filteredNodes = allKGData.nodes.filter(node => nodeIds.has(node.id));
  
  if (filteredNodes.length === 0) {
    container.innerHTML = '<div class="graph-placeholder"><p>No matching graph relations found for this claim.</p></div>';
    return;
  }
  
  const nodes = new vis.DataSet(filteredNodes.map(n => {
    const isPrimary = entitySet.has(n.id.toLowerCase());
    return {
      id: n.id,
      label: n.id,
      shape: 'dot',
      size: isPrimary ? 22 : 14,
      font: { color: '#f0f6fc', size: isPrimary ? 12 : 10 },
      borderWidth: isPrimary ? 2 : 0,
      color: isPrimary
        ? { background: '#388bfd', border: '#58a6ff' }
        : { background: '#21262d', border: '#30363d' }
    };
  }));
  
  const edges = new vis.DataSet(filteredEdges.map(e => ({
    from: e.from,
    to: e.to,
    label: e.label,
    font: { color: '#8b949e', size: 8, align: 'middle' },
    arrows: 'to',
    color: { color: 'rgba(56, 139, 253, 0.4)', highlight: '#388bfd' },
    smooth: { type: 'curvedCW', roundness: 0.15 }
  })));
  
  const options = {
    physics: { stabilization: { iterations: 60 }, solver: 'forceAtlas2Based' },
    interaction: { hover: true, zoomView: true, dragView: true }
  };
  
  new vis.Network(container, { nodes, edges }, options);
}

/* 3. Knowledge Graph Explorer */
function setupKGExplorer() {
  document.getElementById('btn-rebuild-kg').addEventListener('click', async () => {
    if (!confirm('Rebuild entire Knowledge Graph from source Wiki files?')) return;
    const btn = document.getElementById('btn-rebuild-kg');
    btn.disabled = true;
    btn.textContent = 'Rebuilding...';
    try {
      const res = await fetch('/api/kg-manage/rebuild', { method: 'POST' });
      const data = await res.json();
      if (data.success) {
        alert(`KG successfully rebuilt! Loaded ${data.triples} triples.`);
        loadFullKG();
      } else {
        throw new Error(data.error);
      }
    } catch (err) {
      alert('Failed to rebuild KG: ' + err.message);
    } finally {
      btn.disabled = false;
      btn.textContent = 'Rebuild KG from Wiki';
    }
  });

  document.getElementById('btn-clear-kg').addEventListener('click', async () => {
    if (!confirm('Are you absolutely sure you want to clear the entire local Knowledge Graph? This cannot be undone.')) return;
    try {
      const res = await fetch('/api/kg-manage/clear', { method: 'POST' });
      const data = await res.json();
      if (data.success) {
        alert('Knowledge Graph cleared.');
        loadFullKG();
      } else {
        throw new Error(data.error);
      }
    } catch (err) {
      alert('Failed to clear KG: ' + err.message);
    }
  });

  document.getElementById('add-triple-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const form = e.target;
    const fd = new FormData(form);
    try {
      const res = await fetch('/api/kg-manage/add-triple', { method: 'POST', body: fd });
      const data = await res.json();
      if (data.success) {
        alert('Triple added successfully!');
        form.reset();
        loadFullKG();
      } else {
        throw new Error(data.error);
      }
    } catch (err) {
      alert('Error adding relation: ' + err.message);
    }
  });

  document.getElementById('btn-search-kg').addEventListener('click', searchKGNode);
  document.getElementById('search-kg-input').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') searchKGNode();
  });
  
  document.getElementById('btn-reset-kg').addEventListener('click', () => {
    if (fullKGNetwork) {
      fullKGNetwork.fit({ animation: true });
    }
  });

  document.getElementById('btn-toggle-kg-full').addEventListener('click', () => {
    fullKGMode = fullKGMode === 'all' ? 'communities' : 'all';
    selectedKGCommunity = null;
    renderFullKGGraph();
  });

  document.getElementById('btn-close-preview').addEventListener('click', () => {
    document.getElementById('wiki-preview-panel').classList.add('hidden');
    document.querySelectorAll('#wiki-pages-table tbody tr').forEach(r => r.classList.remove('selected'));
  });
}

async function loadFullKG() {
  const container = document.getElementById('full-kg-visualization');
  container.innerHTML = '<div class="graph-placeholder"><span class="spinner"></span><p style="margin-top:10px">Loading Knowledge Graph...</p></div>';
  
  try {
    const res = await fetch('/api/kg-data');
    if (!res.ok) throw new Error('Failed to load KG API');
    const data = await res.json();
    fullKGData = data; // Cache globally
    fullKGMode = 'communities';
    selectedKGCommunity = null;
    
    document.getElementById('full-kg-stats').textContent = 
      `${data.nodes.length} nodes · ${data.edges.length} edges · ${data.stats.pages} articles · ${data.stats.total_triples} total triples`;
    
    if (!data.nodes.length) {
      container.innerHTML = '<div class="graph-placeholder"><p>Knowledge Graph is currently empty. Click Rebuild KG to populate.</p></div>';
      return;
    }

    renderFullKGGraph();
    return;

  } catch (err) {
    container.innerHTML = `<div class="graph-placeholder"><p style="color:var(--red)">Failed to load graph: ${err.message}</p></div>`;
  }
}

function buildKGCommunities(data) {
  const communities = new Map();
  const nodeToCommunities = new Map();
  const communityMeta = new Map((data.communities || []).map(item => [item.id, item]));

  data.edges.forEach(edge => {
    const source = edge.community_id || edge.source || 'Unknown source';
    const meta = communityMeta.get(source);
    if (!communities.has(source)) {
      communities.set(source, {
        id: source,
        label: edge.community_label || (meta && meta.label) || source.replace(/\.md$/i, '').replace(/-/g, ' '),
        nodes: new Set(),
        edges: []
      });
    }

    const community = communities.get(source);
    community.nodes.add(edge.from);
    community.nodes.add(edge.to);
    community.edges.push(edge);

    [edge.from, edge.to].forEach(nodeId => {
      if (!nodeToCommunities.has(nodeId)) nodeToCommunities.set(nodeId, new Set());
      nodeToCommunities.get(nodeId).add(source);
    });
  });

  data.nodes.forEach(node => {
    if (!nodeToCommunities.has(node.id)) {
      const source = 'Unlinked entities';
      if (!communities.has(source)) {
        communities.set(source, { id: source, label: source, nodes: new Set(), edges: [] });
      }
      communities.get(source).nodes.add(node.id);
      nodeToCommunities.set(node.id, new Set([source]));
    }
  });

  return { communities: [...communities.values()], nodeToCommunities };
}

function makeKGEntityNode(node) {
  return {
    id: node.id,
    label: node.label,
    shape: 'dot',
    size: 15,
    font: { color: '#c9d1d9', size: 11 },
    borderWidth: 0,
    color: node.group === 1
      ? { background: '#388bfd', border: '#388bfd' }
      : { background: '#3fb950', border: '#3fb950' }
  };
}

function makeKGEdge(edge) {
  return {
    from: edge.from,
    to: edge.to,
    label: edge.label,
    title: edge.title || '',
    font: { color: '#8b949e', size: 9, align: 'middle' },
    arrows: 'to',
    color: { color: 'rgba(48, 54, 61, 0.75)', highlight: '#388bfd' },
    smooth: { type: 'curvedCW', roundness: 0.15 }
  };
}

function getKGNetworkOptions(iterations, gravity, springLength) {
  return {
    physics: {
      stabilization: { iterations },
      solver: 'forceAtlas2Based',
      forceAtlas2Based: {
        gravitationalConstant: gravity,
        centralGravity: 0.005,
        springLength,
        avoidOverlap: 0.9
      },
      minVelocity: 0.75
    },
    interaction: { hover: true, tooltipDelay: 200, zoomView: true, dragView: true },
    layout: { improvedLayout: false }
  };
}

function getKGCommunityLabel(label, nodeCount) {
  const clean = label.length > 34 ? `${label.slice(0, 32)}...` : label;
  return `${clean} (${nodeCount})`;
}

function getKGCommunitySummary(community) {
  const degree = new Map();
  community.edges.forEach(edge => {
    degree.set(edge.from, (degree.get(edge.from) || 0) + 1);
    degree.set(edge.to, (degree.get(edge.to) || 0) + 1);
  });

  return [...degree.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 4)
    .map(([node]) => node);
}

function renderKGCommunityOverview(communities) {
  const container = document.getElementById('full-kg-visualization');
  const sortedCommunities = [...communities].sort((a, b) => b.nodes.size - a.nodes.size);

  container.innerHTML = `
    <div class="kg-community-overview">
      ${sortedCommunities.map((community, index) => {
        const topNodes = getKGCommunitySummary(community);
        const accent = index % 2 ? 'green' : 'blue';
        return `
          <button class="kg-community-card ${accent}" type="button" data-community-id="${escapeHtml(community.id)}" aria-label="Open ${escapeHtml(community.label)} community">
            <span class="kg-community-orb">${community.nodes.size}</span>
            <span class="kg-community-content">
              <span class="kg-community-title">${escapeHtml(community.label.replace(/\.md$/i, ''))}</span>
              <span class="kg-community-meta">${community.edges.length} relations</span>
              <span class="kg-community-preview">${topNodes.map(escapeHtml).join(' · ') || 'No connected entities'}</span>
            </span>
          </button>
        `;
      }).join('')}
    </div>
  `;

  container.querySelectorAll('.kg-community-card').forEach(card => {
    card.addEventListener('click', () => {
      selectedKGCommunity = card.dataset.communityId;
      renderFullKGGraph();
    });
  });
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

function renderKGCommunitySubgraph(community, nodeById) {
  const container = document.getElementById('full-kg-visualization');
  container.innerHTML = `
    <div class="kg-subgraph-toolbar">
      <button class="btn btn-secondary btn-sm" id="btn-back-kg-communities" type="button">Back to communities</button>
      <span>${escapeHtml(getKGCommunityLabel(community.label, community.nodes.size))}</span>
    </div>
    <div id="kg-community-network" class="kg-community-network"></div>
  `;

  document.getElementById('btn-back-kg-communities').addEventListener('click', () => {
    selectedKGCommunity = null;
    renderFullKGGraph();
  });

  const communityNodes = [...community.nodes]
    .map(nodeId => nodeById.get(nodeId))
    .filter(Boolean)
    .map(makeKGEntityNode);
  const communityEdges = community.edges.map(makeKGEdge);
  const networkEl = document.getElementById('kg-community-network');

  fullKGNetwork = new vis.Network(
    networkEl,
    { nodes: new vis.DataSet(communityNodes), edges: new vis.DataSet(communityEdges) },
    getKGNetworkOptions(90, -45, 95)
  );

  fullKGNetwork.on("stabilizationIterationsDone", () => fullKGNetwork.fit({ animation: true }));
}

function renderFullKGGraph() {
  if (!fullKGData) return;

  const container = document.getElementById('full-kg-visualization');
  const stats = document.getElementById('full-kg-stats');
  const toggleBtn = document.getElementById('btn-toggle-kg-full');

  if (fullKGNetwork) {
    fullKGNetwork.destroy();
    fullKGNetwork = null;
  }
  container.innerHTML = '';

  if (fullKGMode === 'all') {
    const nodes = new vis.DataSet(fullKGData.nodes.map(makeKGEntityNode));
    const edges = new vis.DataSet(fullKGData.edges.map(makeKGEdge));

    fullKGNetwork = new vis.Network(container, { nodes, edges }, getKGNetworkOptions(120, -35, 80));
    fullKGNetwork.on("stabilizationIterationsDone", () => fullKGNetwork.fit());
    stats.textContent = `${fullKGData.nodes.length} nodes - ${fullKGData.edges.length} edges - full graph view`;
    toggleBtn.textContent = 'Community view';
    return;
  }

  const { communities } = buildKGCommunities(fullKGData);
  const nodeById = new Map(fullKGData.nodes.map(node => [node.id, node]));

  if (selectedKGCommunity) {
    const community = communities.find(item => item.id === selectedKGCommunity);
    if (community) {
      renderKGCommunitySubgraph(community, nodeById);
      stats.textContent = `${community.nodes.size} nodes - ${community.edges.length} edges - ${community.label}`;
      toggleBtn.textContent = 'Show all nodes';
      return;
    }
    selectedKGCommunity = null;
  }

  renderKGCommunityOverview(communities);
  stats.textContent = `${communities.length} communities - ${fullKGData.nodes.length} total nodes - ${fullKGData.edges.length} edges`;
  toggleBtn.textContent = 'Show all nodes';
}

function searchKGNode() {
  const q = document.getElementById('search-kg-input').value.trim().toLowerCase();
  if (!q || !fullKGNetwork || !fullKGData) return;
  
  // Find node matching query
  const matched = fullKGData.nodes.find(n => n.id.toLowerCase().includes(q));
  if (matched) {
    if (fullKGMode === 'communities') {
      const { nodeToCommunities } = buildKGCommunities(fullKGData);
      const firstCommunity = [...(nodeToCommunities.get(matched.id) || [])][0];
      if (firstCommunity) {
        selectedKGCommunity = firstCommunity;
        renderFullKGGraph();
      }
    }

    fullKGNetwork.selectNodes([matched.id]);
    fullKGNetwork.focus(matched.id, {
      scale: 1.1,
      animation: { duration: 600, easingFunction: 'easeInOutQuad' }
    });
  } else {
    alert(`Entity node containing "${q}" was not found in the KG.`);
  }
}

async function loadWikiPages() {
  const tbody = document.querySelector('#wiki-pages-table tbody');
  tbody.innerHTML = '<tr><td colspan="3" class="text-muted" style="text-align:center">Scanning articles...</td></tr>';
  
  try {
    const res = await fetch('/api/wiki-pages');
    const pages = await res.json();
    tbody.innerHTML = '';
    
    if (!pages.length) {
      tbody.innerHTML = '<tr><td colspan="3" class="text-muted" style="text-align:center">No compiled Wiki articles found.</td></tr>';
      return;
    }
    
    pages.forEach(p => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td style="font-weight:600; color:#fff">${p.title}</td>
        <td style="font-family:monospace; color:var(--text-muted)">${p.filename}</td>
        <td>${(p.size / 1024).toFixed(1)} KB</td>
      `;
      tr.addEventListener('click', () => {
        document.querySelectorAll('#wiki-pages-table tbody tr').forEach(r => r.classList.remove('selected'));
        tr.classList.add('selected');
        loadWikiPagePreview(p.filename);
      });
      tbody.appendChild(tr);
    });
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="3" style="color:var(--red)">Error loading pages: ${err.message}</td></tr>`;
  }
}

async function loadWikiPagePreview(filename) {
  const panel = document.getElementById('wiki-preview-panel');
  panel.classList.remove('hidden');
  
  document.getElementById('preview-title').textContent = 'Loading...';
  document.getElementById('preview-summary').textContent = '';
  document.getElementById('preview-entities').innerHTML = '';
  document.getElementById('preview-relations').innerHTML = '';
  
  try {
    const res = await fetch(`/api/wiki-pages/${filename}`);
    if (!res.ok) throw new Error('API Error');
    const data = await res.json();
    
    document.getElementById('preview-title').textContent = data.title;
    document.getElementById('preview-summary').textContent = data.summary || 'No summary available.';
    
    // Entities tags
    const entDiv = document.getElementById('preview-entities');
    if (data.entities && data.entities.length) {
      data.entities.forEach(e => {
        const span = document.createElement('span');
        span.textContent = e;
        entDiv.appendChild(span);
      });
    } else {
      entDiv.innerHTML = '<span class="text-muted">No entities cached.</span>';
    }
    
    // Relations list
    const relUl = document.getElementById('preview-relations');
    if (data.relations && data.relations.length) {
      data.relations.forEach(r => {
        const li = document.createElement('li');
        li.textContent = `(${r.subject} -> ${r.predicate} -> ${r.object})`;
        relUl.appendChild(li);
      });
    } else {
      relUl.innerHTML = '<li class="text-muted" style="background:none; border:none; padding:0">No relations stored.</li>';
    }
  } catch (err) {
    document.getElementById('preview-title').textContent = 'Failed to load preview';
  }
}

function renderNliPremise(data) {
  const box = document.getElementById('premise-box');
  box.innerHTML = '';
  const decisionEvidence = data.nli_decision_evidence || [];

  if (decisionEvidence.length) {
    const title = document.createElement('div');
    title.className = 'decision-title';
    title.textContent = 'Evidence that drove the verdict';
    box.appendChild(title);

    decisionEvidence.forEach((item, idx) => {
      const row = document.createElement('div');
      row.className = 'decision-evidence-item';

      const meta = document.createElement('div');
      meta.className = 'decision-evidence-meta';
      meta.textContent = `${idx + 1}. ${item.verdict} | confidence ${(item.confidence * 100).toFixed(1)}% | weight ${Number(item.effective_weight || 0).toFixed(3)}`;

      const text = document.createElement('div');
      text.className = 'decision-evidence-text';
      text.textContent = item.text || 'No evidence text available.';

      if (item.source) {
        const source = document.createElement('div');
        source.className = 'decision-evidence-source';
        source.textContent = `Source: ${item.source_type || 'evidence'} / ${item.source}`;
        row.append(meta, text, source);
      } else {
        row.append(meta, text);
      }
      box.appendChild(row);
    });
  }

  const premiseTitle = document.createElement('div');
  premiseTitle.className = 'decision-title';
  premiseTitle.textContent = decisionEvidence.length ? 'Full NLI premise' : 'NLI premise';
  box.appendChild(premiseTitle);

  const premise = document.createElement('div');
  premise.className = 'decision-premise-text';
  premise.textContent = data.premise || 'No evidence gathered.';
  box.appendChild(premise);
}

/* 4. Wiki Explorer */
function setupWikiExplorer() {
  document.getElementById('wiki-reader-search').addEventListener('input', renderWikiReaderPages);
}

async function loadWikiReaderPages() {
  const list = document.getElementById('wiki-reader-pages');
  list.innerHTML = '<div class="text-muted">Scanning articles...</div>';

  try {
    const res = await fetch('/api/wiki-pages');
    wikiReaderPages = await res.json();
    renderWikiReaderPages();
  } catch (err) {
    list.innerHTML = `<div style="color:var(--red)">Error loading pages: ${err.message}</div>`;
  }
}

function renderWikiReaderPages() {
  const q = document.getElementById('wiki-reader-search').value.trim().toLowerCase();
  const list = document.getElementById('wiki-reader-pages');
  const pages = wikiReaderPages.filter(p =>
    p.title.toLowerCase().includes(q) || p.filename.toLowerCase().includes(q)
  );

  if (!pages.length) {
    list.innerHTML = '<div class="text-muted">No wiki pages found.</div>';
    return;
  }

  list.innerHTML = '';
  pages.forEach(p => {
    const btn = document.createElement('button');
    const title = document.createElement('strong');
    const file = document.createElement('span');
    btn.type = 'button';
    btn.className = 'wiki-reader-page';
    title.textContent = p.title;
    file.textContent = p.filename;
    btn.append(title, file);
    btn.addEventListener('click', () => loadWikiReaderPage(p.filename, btn));
    list.appendChild(btn);
  });
}

async function loadWikiReaderPage(filename, btn) {
  document.querySelectorAll('.wiki-reader-page').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById('wiki-reader-title').textContent = 'Loading...';
  document.getElementById('wiki-reader-filename').textContent = filename;
  document.getElementById('wiki-reader-summary').textContent = '';
  document.getElementById('wiki-reader-graph').innerHTML = '';

  try {
    const res = await fetch(`/api/wiki-pages/${filename}`);
    if (!res.ok) throw new Error('API Error');
    const data = await res.json();
    document.getElementById('wiki-reader-title').textContent = data.title;
    document.getElementById('wiki-reader-summary').textContent = data.summary || 'No summary available.';
    renderWikiReaderGraph(data.relations || []);
  } catch (err) {
    document.getElementById('wiki-reader-title').textContent = 'Failed to load page';
    document.getElementById('wiki-reader-summary').textContent = err.message;
  }
}

function renderWikiReaderGraph(relations) {
  const container = document.getElementById('wiki-reader-graph');
  if (wikiReaderNetwork) {
    wikiReaderNetwork.destroy();
    wikiReaderNetwork = null;
  }

  if (!relations.length) {
    container.innerHTML = '<div class="graph-placeholder"><p>No relations stored for this page.</p></div>';
    return;
  }

  const nodeIds = new Set();
  relations.forEach(r => {
    nodeIds.add(r.subject);
    nodeIds.add(r.object);
  });

  const nodes = new vis.DataSet([...nodeIds].map(id => ({
    id,
    label: id,
    shape: 'dot',
    size: 16,
    font: { color: '#c9d1d9', size: 11 },
    color: { background: '#388bfd', border: '#58a6ff' }
  })));

  const edges = new vis.DataSet(relations.map(r => ({
    from: r.subject,
    to: r.object,
    label: r.predicate,
    arrows: 'to',
    font: { color: '#8b949e', size: 9, align: 'middle' },
    color: { color: 'rgba(56, 139, 253, 0.45)', highlight: '#388bfd' },
    smooth: { type: 'curvedCW', roundness: 0.15 }
  })));

  wikiReaderNetwork = new vis.Network(container, { nodes, edges }, {
    physics: { stabilization: { iterations: 80 }, solver: 'forceAtlas2Based' },
    interaction: { hover: true, zoomView: true, dragView: true }
  });
}

/* 5. Benchmark Center */
function setupBenchmarkCenter() {
  document.getElementById('benchmark-run-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const btn = document.getElementById('btn-run-benchmark');
    btn.disabled = true;
    
    const fd = new FormData(e.target);
    try {
      const res = await fetch('/api/benchmark/run', { method: 'POST', body: fd });
      const data = await res.json();
      if (data.success) {
        startPollingBenchmarkProgress();
      } else {
        throw new Error(data.error);
      }
    } catch (err) {
      alert('Error launching benchmark: ' + err.message);
      btn.disabled = false;
    }
  });

  document.getElementById('bench-run-select').addEventListener('change', (e) => {
    const fn = e.target.value;
    if (fn) {
      loadBenchmarkDetails(fn);
    } else {
      document.getElementById('benchmark-metrics-display').classList.add('hidden');
      document.getElementById('benchmark-empty-state').classList.remove('hidden');
    }
  });
}

async function loadBenchmarkRunsList() {
  const select = document.getElementById('bench-run-select');
  select.innerHTML = '<option value="">Scanning completed runs...</option>';
  
  try {
    const res = await fetch('/api/benchmark/list');
    const runs = await res.json();
    select.innerHTML = '';
    
    if (!runs.length) {
      select.innerHTML = '<option value="">No completed runs found</option>';
      return;
    }
    
    select.innerHTML = '<option value="">-- Choose past run --</option>';
    runs.forEach(r => {
      const opt = document.createElement('option');
      opt.value = r.filename;
      
      const ts = r.timestamp ? new Date(r.timestamp).toLocaleString() : 'Unknown';
      opt.textContent = `Samples: ${r.samples} | Acc: ${r.accuracy.toFixed(1)}% | (${ts})`;
      select.appendChild(opt);
    });
  } catch (err) {
    select.innerHTML = '<option value="">Error fetching runs</option>';
  }
}

async function loadBenchmarkDetails(filename) {
  try {
    const res = await fetch(`/api/benchmark/details/${filename}`);
    if (!res.ok) throw new Error('API Error');
    const data = await res.json();
    
    const m = data.metrics || {};
    
    // Display panels
    document.getElementById('benchmark-empty-state').classList.add('hidden');
    document.getElementById('benchmark-metrics-display').classList.remove('hidden');
    
    // Render metrics
    document.getElementById('metrics-accuracy').textContent = `${(m.accuracy_pct || 0).toFixed(1)}%`;
    document.getElementById('metrics-precision').textContent = `${(m.precision_pct || 0).toFixed(1)}%`;
    document.getElementById('metrics-recall').textContent = `${(m.recall_pct || 0).toFixed(1)}%`;
    document.getElementById('metrics-f1').textContent = (m.f1_score || 0).toFixed(2);
    
    // Confusion cell counts
    const conf = m.confusion || {};
    document.getElementById('cell-tp').textContent = `${conf.TP || 0} (TP)`;
    document.getElementById('cell-fn').textContent = `${conf.FN || 0} (FN)`;
    document.getElementById('cell-fp').textContent = `${conf.FP || 0} (FP)`;
    document.getElementById('cell-tn').textContent = `${conf.TN || 0} (TN)`;
    
    // Details
    const dt = m.timestamp ? new Date(m.timestamp).toLocaleString() : 'N/A';
    document.getElementById('bench-run-time').textContent = dt;
    document.getElementById('bench-run-total').textContent = m.num_samples || 0;
    document.getElementById('bench-run-errors').textContent = m.errors || 0;
    
    // Error list
    const errList = document.getElementById('bench-error-list');
    errList.innerHTML = '';
    if (data.errors && data.errors.length) {
      data.errors.forEach(err => {
        const item = document.createElement('div');
        item.className = 'error-log-item';
        item.innerHTML = `
          <div class="claim" style="color:#fff">${err.claim}</div>
          <div class="meta-row">
            <span>Gold label: <span class="gold">${err.gold}</span></span>
            <span>Verdict: <span class="pred">${err.pred}</span></span>
            <span>Confidence: ${( (err.probs[err.pred] || 0) * 100 ).toFixed(1)}%</span>
          </div>
          <div class="premise" title="${err.premise}">Premise: ${err.premise}</div>
        `;
        errList.appendChild(item);
      });
    } else {
      errList.innerHTML = '<div class="text-muted" style="padding:1.5rem; text-align:center">No error cases in this run. 100% accuracy achieved!</div>';
    }
    
  } catch (err) {
    alert('Failed to load benchmark details: ' + err.message);
  }
}

async function checkActiveBenchmark() {
  try {
    const res = await fetch('/api/benchmark/status');
    const data = await res.json();
    if (data.status === 'running') {
      startPollingBenchmarkProgress();
    }
  } catch (err) {}
}

function startPollingBenchmarkProgress() {
  document.getElementById('btn-run-benchmark').disabled = true;
  document.getElementById('bench-progress-panel').classList.remove('hidden');
  
  if (activeBenchmarkInterval) clearInterval(activeBenchmarkInterval);
  
  activeBenchmarkInterval = setInterval(async () => {
    try {
      const res = await fetch('/api/benchmark/status');
      const data = await res.json();
      
      const pct = data.total > 0 ? (data.current / data.total) * 100 : 0;
      
      document.getElementById('bench-status-label').textContent = data.status.toUpperCase() + '...';
      document.getElementById('bench-progress-fraction').textContent = `${data.current}/${data.total}`;
      document.getElementById('bench-progress-bar').style.width = pct + '%';
      
      if (data.current_claim) {
        document.getElementById('bench-current-sample').innerHTML = `
          <strong>Current statement:</strong><br>
          <span style="color:#fff">${data.current_claim.substring(0, 150)}...</span>
        `;
      }
      
      // Update logs
      const logDiv = document.getElementById('bench-live-logs');
      logDiv.innerHTML = data.logs.map(l => `<div>${l}</div>`).join('');
      logDiv.scrollTop = logDiv.scrollHeight; // Scroll to bottom
      
      if (data.status === 'completed' || data.status === 'error') {
        clearInterval(activeBenchmarkInterval);
        document.getElementById('btn-run-benchmark').disabled = false;
        loadBenchmarkRunsList();
        
        if (data.status === 'completed') {
          // Find the last completed file from list and load it
          setTimeout(async () => {
            const listRes = await fetch('/api/benchmark/list');
            const runs = await listRes.json();
            if (runs.length) {
              document.getElementById('bench-run-select').value = runs[0].filename;
              loadBenchmarkDetails(runs[0].filename);
            }
          }, 1000);
        } else {
          alert('Benchmark Error: ' + data.error_message);
        }
      }
    } catch (err) {
      clearInterval(activeBenchmarkInterval);
      document.getElementById('btn-run-benchmark').disabled = false;
    }
  }, 1500);
}

/* 6. NER Playground */
function setupNERPlayground() {
  document.getElementById('ner-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const btn = document.getElementById('btn-process-ner');
    const spinner = btn.querySelector('.spinner');
    const btnText = btn.querySelector('.btn-text');
    
    btn.disabled = true;
    spinner.classList.remove('hidden');
    btnText.textContent = 'Analyzing...';
    
    document.getElementById('ner-empty-state').classList.add('hidden');
    document.getElementById('ner-results-display').classList.add('hidden');
    
    const fd = new FormData(e.target);
    try {
      const res = await fetch('/api/extract-entities', { method: 'POST', body: fd });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'API Error');
      
      showNERResults(data, document.getElementById('ner-text').value);
    } catch (err) {
      alert('Playground Error: ' + err.message);
    } finally {
      btn.disabled = false;
      spinner.classList.add('hidden');
      btnText.textContent = 'Extract Entities';
    }
  });
}

function showNERResults(data, text) {
  document.getElementById('ner-results-display').classList.remove('hidden');
  
  // 1. Render highlighted text
  const box = document.getElementById('ner-highlights');
  box.innerHTML = renderHighlightedText(text, data.entities || []);
  if (data.warning) {
    box.innerHTML += `<div class="ner-warning">${escapeHtml(data.warning)}</div>`;
  }
  
  // 2. Populate Noun chunks
  const tbody = document.querySelector('#ner-chunks-table tbody');
  tbody.innerHTML = '';
  
  if (data.noun_chunks && data.noun_chunks.length) {
    data.noun_chunks.forEach(c => {
      const tr = document.createElement('tr');
      const isCand = c.is_entity_candidate;
      const scoreEntries = Object.entries(c.scores || {})
        .sort((a, b) => b[1] - a[1])
        .slice(0, 3);
      const scoreBadges = scoreEntries.length ? scoreEntries.map(([label, score]) => `
        <span class="score-pill" title="${escapeHtml(label)} similarity: ${Number(score).toFixed(3)}. This compares the chunk meaning with the ${escapeHtml(label)} prototype.">
          ${escapeHtml(label)} ${Number(score).toFixed(3)}
        </span>
      `).join('') : '<span class="score-pill" title="This chunk was skipped before SBERT scoring.">not scored</span>';
      const threshold = c.threshold == null ? null : Number(c.threshold);
      const bestScore = Number(c.best_score ?? 0);
      const candidateTitle = threshold == null
        ? 'This chunk was skipped before SBERT scoring.'
        : `Best score ${bestScore.toFixed(3)} ${isCand ? 'meets' : 'does not meet'} threshold ${threshold.toFixed(3)}.`;
      
      tr.innerHTML = `
        <td style="font-weight:600; color:#fff">${escapeHtml(c.text)}</td>
        <td><div class="score-pill-row">${scoreBadges}</div></td>
        <td><span class="score-pill" title="Minimum score needed for this chunk to become an entity candidate.">${threshold == null ? 'n/a' : threshold.toFixed(3)}</span></td>
        <td><span class="badge ${c.best_type === 'PERSON' ? 'badge-info' : 'badge-success'}" title="The prototype with the highest similarity score.">${escapeHtml(c.best_type || 'n/a')}</span></td>
        <td>
          <span class="badge ${isCand ? 'badge-success' : 'badge-danger'}" 
                title="${escapeHtml(candidateTitle)}"
                style="${!isCand ? 'background-color:rgba(248,81,73,0.06); color:var(--red); border:1px solid rgba(248,81,73,0.15)' : ''}">
            ${isCand ? 'YES' : 'NO'} (${(bestScore * 100).toFixed(0)}%)
          </span>
        </td>
      `;
      tbody.appendChild(tr);
    });
  } else {
    tbody.innerHTML = '<tr><td colspan="5" class="text-muted" style="text-align:center">No noun chunks extracted.</td></tr>';
  }
}

function renderHighlightedText(text, entities) {
  const sorted = [...entities].sort((a, b) => a.start - b.start);
  let html = '';
  let cursor = 0;
  sorted.forEach(ent => {
    const start = ent.start;
    const end = ent.end;
    const label = ent.label.toLowerCase();
    if (start < cursor) return;
    
    // Map labels to highlight categories
    let cls = 'misc';
    if (label === 'person') cls = 'person';
    else if (label === 'org') cls = 'org';
    else if (label === 'gpe') cls = 'gpe';
    
    html += escapeHtml(text.substring(cursor, start));
    html += `<span class="ner-highlight ${cls}">${escapeHtml(text.substring(start, end))}<span class="ner-type">${escapeHtml(ent.label)}</span></span>`;
    cursor = end;
  });
  html += escapeHtml(text.substring(cursor));
  return html;
}
