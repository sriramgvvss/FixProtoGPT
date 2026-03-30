/**
 * FixProtoGPT — Conversations Sidebar Module
 * Loads, filters, renders and manages conversation history.
 * Depends on: api.js, ui.js
 * @module conversations
 */

/** @type {Array<Object>} All visible conversations (filtered by hidden IDs). */
let allConversations = [];

/** @type {string} Current filter: 'all'|'generate'|'explain'|'validate'|'complete'. */
let convFilter = 'all';

// ── Sidebar toggle ──────────────────────────────────────────────

function toggleSidebar() {
    const sidebar = document.getElementById('convSidebar');
    const overlay = document.getElementById('convOverlay');
    if (!sidebar) return;
    sidebar.classList.toggle('open');
    if (overlay) overlay.classList.toggle('open');
}

// ── Hidden-IDs (localStorage) ───────────────────────────────────

function getHiddenConvIds() {
    try { return JSON.parse(localStorage.getItem('hiddenConvIds') || '[]'); }
    catch { return []; }
}

function addHiddenConvIds(ids) {
    const hidden = new Set(getHiddenConvIds());
    ids.forEach(id => hidden.add(id));
    localStorage.setItem('hiddenConvIds', JSON.stringify([...hidden]));
}

// ── Load & render ───────────────────────────────────────────────

async function loadConversations() {
    const list = document.getElementById('convList');
    const loading = document.getElementById('convLoading');
    if (!list || !loading) return;

    loading.style.display = 'block';
    list.innerHTML = '';

    try {
        const resp = await authFetch(`${API_BASE}/interactions`);
        const data = await resp.json();
        const hidden = new Set(getHiddenConvIds());
        allConversations = (data.interactions || [])
            .map(c => ({ ...c, interaction_id: c.interaction_id || c.id }))
            .filter(c => !hidden.has(c.interaction_id))
            .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
        renderConvStats(data);
        renderConversations();
    } catch (err) {
        list.innerHTML = '<div class="alert alert-danger">Failed to load conversations.</div>';
        console.error('loadConversations:', err);
    } finally {
        loading.style.display = 'none';
    }
}

function renderConvStats() {
    const el = document.getElementById('convStats');
    if (!el) return;
    const total = allConversations.length;
    el.innerHTML = '<span><strong>' + total + '</strong> interactions</span>';
}

function renderConversations() {
    const list = document.getElementById('convList');
    if (!list) return;

    const filtered = _applyFilter(allConversations, convFilter);

    if (filtered.length === 0) {
        list.innerHTML =
            '<div class="conv-empty">' +
            '<i class="fas fa-inbox"></i>' +
            '<div>No conversations yet.</div>' +
            '</div>';
        return;
    }

    list.innerHTML = filtered.map(c => _buildConvCard(c)).join('');
}

// ── Filtering ───────────────────────────────────────────────────

function filterConversations(filter, btn) {
    convFilter = filter;
    document.querySelectorAll('.conv-filter-bar .btn').forEach(b => b.classList.remove('active'));
    if (btn) btn.classList.add('active');
    renderConversations();
}

// ── Actions ─────────────────────────────────────────────────────

function copyConversation(id) {
    const conv = allConversations.find(c => c.interaction_id === id);
    if (!conv) return;
    const text = 'Request: ' + _summariseRequest(conv.request) + '\nResponse: ' + _summariseResponse(conv.response);
    navigator.clipboard.writeText(text).then(() => {
        showToast('📋 Copied to clipboard!', 'success');
    }).catch(() => {
        showToast('❌ Copy failed', 'error');
    });
}

function deleteConversation(id) {
    if (!confirm('Delete this conversation?')) return;
    addHiddenConvIds([id]);
    const card = document.getElementById('conv-' + id);
    if (card) {
        card.style.transition = 'all 0.3s ease';
        card.style.opacity = '0';
        card.style.transform = 'translateX(60px)';
        setTimeout(() => card.remove(), 300);
    }
    allConversations = allConversations.filter(c => c.interaction_id !== id);
    renderConvStats();
    showToast('🗑️ Conversation hidden', 'success');
}

function clearAllConversations() {
    if (!confirm('Clear all conversations from view?')) return;
    addHiddenConvIds(allConversations.map(c => c.interaction_id));
    allConversations = [];
    renderConvStats();
    renderConversations();
    showToast('🗑️ All conversations cleared', 'success');
}

function downloadConversations() {
    if (!allConversations.length) {
        showToast('⚠️ No conversations to download', 'error');
        return;
    }

    const filtered = _applyFilter(allConversations, convFilter);

    const payload = {
        exported_at: new Date().toISOString(),
        filter: convFilter,
        count: filtered.length,
        conversations: filtered.map(c => ({
            id: c.id,
            endpoint: c.endpoint,
            timestamp: c.timestamp,
            request: c.request,
            response: c.response,
            feedback: c.feedback || null
        }))
    };

    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'fixprotogpt_chats_' + new Date().toISOString().slice(0, 10) + '.json';
    document.body.appendChild(a);
    a.click();
    URL.revokeObjectURL(url);
    document.body.removeChild(a);
    showToast('📥 Downloaded ' + filtered.length + ' conversation(s)', 'success');
}

// ── Private helpers ─────────────────────────────────────────────

function _getFeedbackRating(c) {
    if (!c.feedback) return null;
    if (typeof c.feedback === 'string') return c.feedback;
    return c.feedback.rating || null;
}

function _applyFilter(conversations, filter) {
    if (filter === 'all') return conversations;
    if (filter === 'generate') {
        return conversations.filter(c => {
            const ep = c.endpoint || '';
            return ep.includes('generate') || ep.includes('nl2fix');
        });
    }
    return conversations.filter(c => (c.endpoint || '').includes(filter));
}

function _buildConvCard(c) {
    const epKey = (c.endpoint || 'generate').replace(/^\/api\//, '');
    const epClasses = {
        nl2fix: 'conv-ep-nl2fix', generate: 'conv-ep-generate',
        explain: 'conv-ep-explain', validate: 'conv-ep-validate',
        complete: 'conv-ep-complete'
    };
    const epClass = epClasses[epKey] || 'conv-ep-generate';
    const epLabel = epKey === 'nl2fix' ? 'Generate' : epKey.charAt(0).toUpperCase() + epKey.slice(1);

    const time = c.timestamp
        ? new Date(c.timestamp).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })
        : '';

    const reqText = _summariseRequest(c.request);
    const resText = _summariseResponse(c.response);

    let feedbackBadge = '';

    return '<div class="conv-card" id="conv-' + c.interaction_id + '">'
        + '<div class="d-flex justify-content-between align-items-center flex-wrap gap-1">'
        + '<span class="conv-endpoint ' + epClass + '">' + epLabel + '</span>'
        + '<span class="conv-time">' + time + '</span>'
        + '</div>'
        + '<div class="conv-request"><strong>Request:</strong> ' + escapeHtml(reqText) + '</div>'
        + '<div class="conv-response" onclick="this.classList.toggle(\'expanded\')">' + escapeHtml(resText) + ' <small style="color:var(--primary);cursor:pointer;">[tap to expand]</small></div>'
        + '<div class="conv-actions">'
        + feedbackBadge
        + '<button class="btn btn-sm btn-outline-secondary" onclick="copyConversation(\'' + c.interaction_id + '\')" title="Copy to clipboard"><i class="fas fa-copy"></i></button>'
        + '<button class="btn btn-sm btn-outline-danger ms-auto" onclick="deleteConversation(\'' + c.interaction_id + '\')"><i class="fas fa-trash-alt"></i></button>'
        + '</div></div>';
}

function _summariseRequest(req) {
    if (!req) return '(no request)';
    if (typeof req === 'string') return req.slice(0, 200);
    if (req.prompt) return req.prompt.slice(0, 200);
    if (req.message) return req.message.slice(0, 200);
    if (req.partial_message) return 'Partial: ' + req.partial_message.slice(0, 160);
    return JSON.stringify(req).slice(0, 200);
}

function _summariseResponse(res) {
    if (!res) return '(no response)';
    if (typeof res === 'string') return res.slice(0, 300);
    if (res.fix_message) return res.fix_message.slice(0, 300);
    if (res.summary) return res.summary.slice(0, 300);
    if (res.generated) return res.generated.slice(0, 300);
    if (res.valid !== undefined) return res.valid ? '✅ Valid' : '❌ Invalid — ' + (res.errors || []).join(', ').slice(0, 200);
    if (res.completed) return res.completed.slice(0, 300);
    return JSON.stringify(res).slice(0, 300);
}
