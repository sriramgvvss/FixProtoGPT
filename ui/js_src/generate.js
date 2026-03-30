/**
 * FixProtoGPT — FIX Generation / Validate / Explain / Complete Module
 * All endpoint interaction logic for the four main tabs.
 * Depends on: api.js, ui.js, conversations.js
 * @module generate
 */

/**
 * Detect whether input looks like a raw FIX-style prompt (contains FIX tags).
 * @param {string} text
 * @returns {boolean}
 */
function isFIXPrompt(text) {
    return /\b8=FIX/i.test(text) || /\|\d+=/.test(text) || /^35=/.test(text);
}

/**
 * Generic endpoint wrapper that handles loading-spinner, error display,
 * and post-render housekeeping (conversations, token badge).
 *
 * @param {Object} opts
 * @param {string} opts.inputId    - DOM id of the text input
 * @param {string} opts.outputId   - DOM id of the output container
 * @param {string} opts.loadingId  - DOM id of the loading spinner
 * @param {string} opts.emptyMsg   - Error message when input is empty
 * @param {Function} opts.apiFn    - async (input) => Response
 * @param {Function} opts.renderFn - (data, outputDiv) => void — renders into outputDiv
 * @param {string} [opts.toast]    - Optional toast message on success
 */
async function _callEndpoint(opts) {
    var input = document.getElementById(opts.inputId).value.trim();
    var outputDiv = document.getElementById(opts.outputId);
    var loadingDiv = document.getElementById(opts.loadingId);

    if (!input) { showError(outputDiv, opts.emptyMsg); return; }

    loadingDiv.style.display = 'block';
    outputDiv.innerHTML = '';

    try {
        var response = await opts.apiFn(input);
        var data = await response.json();
        loadingDiv.style.display = 'none';

        if (data.error) { showError(outputDiv, data.error); return; }

        opts.renderFn(data, outputDiv);
        if (opts.toast) showToast(opts.toast, 'success');
        loadConversations();
        refreshTokenBadge();
    } catch (error) {
        loadingDiv.style.display = 'none';
        showError(outputDiv, 'Error: ' + error.message);
    }
}

// ── Generate FIX Message ────────────────────────────────────────

/**
 * Auto-detect natural language vs FIX prompt and generate a FIX message.
 */
async function generateFix() {
    const input = document.getElementById('generateFixInput').value.trim();
    const outputDiv = document.getElementById('generateFixOutput');
    const loadingDiv = document.getElementById('generateFixLoading');

    if (!input) {
        showError(outputDiv, 'Please enter an instruction or FIX prompt');
        return;
    }

    loadingDiv.style.display = 'block';
    outputDiv.innerHTML = '';

    const useFIXGenerate = isFIXPrompt(input);

    try {
        let response;
        if (useFIXGenerate) {
            response = await authFetch(`${API_BASE}/generate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ prompt: input, temperature: 0.3, max_tokens: 256 })
            });
        } else {
            response = await authFetch(`${API_BASE}/nl2fix`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: input })
            });
        }

        const data = await response.json();
        loadingDiv.style.display = 'none';

        if (data.error) {
            showError(outputDiv, data.error);
            document.getElementById('exportFixJSON').style.display = 'none';
            document.getElementById('exportFixXML').style.display = 'none';
            return;
        }

        const fixMessage = data.fix_message || data.generated;
        const interactionType = useFIXGenerate ? 'generate' : 'nl2fix';
        const isMulti = data.multi_order && data.order_count > 1;

        // Cache for export buttons
        document.getElementById('generatedFixCache').value = fixMessage;
        document.getElementById('exportFixJSON').style.display = 'inline-block';
        document.getElementById('exportFixXML').style.display = 'inline-block';

        let html = '<div class="alert alert-success alert-custom">';
        if (data.demo_mode) {
            html += '<div class="badge bg-warning text-dark mb-3"><i class="fas fa-flask me-1"></i> Demo Mode</div><br>';
            if (data.message) html += '<small>' + escapeHtml(data.message) + '</small><br>';
            if (data.available_models && data.available_models.length) {
                var names = data.available_models.map(function(m) { return m.label; }).join(', ');
                html += '<small class="text-muted">Available models: ' + escapeHtml(names)
                    + '. Use the version selector to switch.</small><br>';
            }
        } else {
            html += '<div class="badge bg-success mb-3"><i class="fas fa-brain me-1"></i> Model Generated</div><br>';
        }
        if (isMulti) {
            html += '<strong><i class="fas fa-bolt me-2"></i>' + data.order_count + ' FIX Messages Generated!</strong></div>';
            const msgs = fixMessage.split('\n').filter(function(m) { return m.trim(); });
            const perOrder = data.order_insights || [];
            msgs.forEach(function(msg, idx) {
                html += '<div class="card border-0 shadow-sm mb-3">';
                html += '<div class="card-body p-3">';
                html += '<div class="d-flex align-items-center mb-2">';
                html += '<span class="badge bg-primary me-2">Order ' + (idx + 1) + '</span>';
                html += '<button class="btn btn-copy btn-sm ms-auto" onclick="copyToClipboard(\'' + escapeHtml(msg).replace(/'/g, "\\'") + '\')"><i class="fas fa-copy me-1"></i>Copy</button>';
                html += '</div>';
                html += '<div class="output-box output-box-compact">' + escapeHtml(msg) + '</div>';
                // Per-order insight
                if (perOrder[idx]) {
                    html += _renderExplainHeader(perOrder[idx]);
                    html += _renderModelInsight(perOrder[idx]);
                }
                html += '</div></div>';
            });
            html += '<button class="btn btn-copy" onclick="copyToClipboard(\'' + escapeHtml(fixMessage).replace(/\n/g, '\\n').replace(/'/g, "\\'") + '\')"><i class="fas fa-copy me-1"></i>Copy All</button>';
        } else {
            html += '<strong><i class="fas fa-bolt me-2"></i>FIX Message Generated!</strong></div>';
            html += '<div class="output-box">' + escapeHtml(fixMessage) + '</div>';
            html += '<button class="btn btn-copy" onclick="copyToClipboard(\'' + escapeHtml(fixMessage).replace(/'/g, "\\'") + '\')"><i class="fas fa-copy"></i> Copy to Clipboard</button>';
            html += _renderExplainHeader(data);
            html += _renderModelInsight(data);
        }
        html += buildFeedbackBar(data.interaction_id, interactionType);
        outputDiv.innerHTML = html;
        showToast(isMulti ? '✨ ' + data.order_count + ' FIX messages generated!' : '✨ FIX message generated!', 'success');
        loadConversations();
        refreshTokenBadge();
    } catch (error) {
        loadingDiv.style.display = 'none';
        showError(outputDiv, 'Error: ' + error.message);
    }
}

// ── Validate FIX Message ────────────────────────────────────────

async function validate() {
    await _callEndpoint({
        inputId: 'validateInput',
        outputId: 'validateOutput',
        loadingId: 'validateLoading',
        emptyMsg: 'Please enter a FIX message',
        apiFn: function(input) {
            return authFetch(API_BASE + '/validate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: input })
            });
        },
        renderFn: function(data, outputDiv) {
            var html = '';
            if (data.valid) {
                html += '<div class="alert alert-success alert-custom">';
                html += '<h5><i class="fas fa-check-circle me-2"></i>✅ Valid FIX Message!</h5>';
                html += '<p class="mb-0"><strong>Fields Found:</strong> ' + data.num_fields + '</p>';
                html += '</div>';
                showToast('✅ Message is valid!', 'success');
            } else {
                html += '<div class="alert alert-danger alert-custom">';
                html += '<h5><i class="fas fa-exclamation-triangle me-2"></i>❌ Invalid FIX Message</h5>';
                html += '<p class="mb-0"><strong>Missing Required Fields:</strong> '
                    + data.missing_required_fields.map(function(f) { return '<code>' + f + '</code>'; }).join(', ') + '</p>';
                html += '</div>';
                showToast('⚠️ Validation failed', 'error');
            }
            if (data.demo_mode && data.available_models && data.available_models.length) {
                var names = data.available_models.map(function(m) { return m.label; }).join(', ');
                html += '<div class="alert alert-warning mt-2 mb-0 py-1 px-2" style="font-size:0.8rem;">'
                    + '<i class="fas fa-info-circle me-1"></i>Demo mode — basic validation only. '
                    + 'Available models: ' + escapeHtml(names) + '</div>';
            }
            if (data.fields && data.fields.length > 0) {
                html += '<h5 class="mt-4 mb-3"><i class="fas fa-list me-2"></i>Field Breakdown</h5>';
                html += '<div style="max-height: 400px; overflow-y: auto;">';
                html += '<table class="table table-sm field-table">';
                html += '<thead><tr><th>Tag</th><th>Name</th><th>Value</th></tr></thead><tbody>';
                data.fields.forEach(function(field) {
                    html += '<tr>';
                    html += '<td><strong>' + escapeHtml(field.tag) + '</strong></td>';
                    html += '<td>' + escapeHtml(field.name) + '</td>';
                    html += '<td><code>' + escapeHtml(field.value) + '</code></td>';
                    html += '</tr>';
                });
                html += '</tbody></table></div>';
            }
            html += _renderExplainHeader(data);
            html += _renderModelInsight(data);
            outputDiv.innerHTML = html;
        }
    });
}

// ── Explain FIX Message ─────────────────────────────────────────

async function explain() {
    await _callEndpoint({
        inputId: 'explainInput',
        outputId: 'explainOutput',
        loadingId: 'explainLoading',
        emptyMsg: 'Please enter a FIX message',
        toast: '📚 Explanation ready!',
        apiFn: function(input) {
            return authFetch(API_BASE + '/explain', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: input })
            });
        },
        renderFn: function(data, outputDiv) {
            var html = '';
            var ex = data.explanation;

            if (ex && typeof ex === 'object' && !Array.isArray(ex)) {
                html += _renderExplainHeader(ex);
                html += _renderExplainSummary(ex);
                html += _renderModelInsight(ex);
                html += _renderFieldBreakdown(ex);
            } else if (typeof ex === 'string') {
                html += '<div class="output-box">' + escapeHtml(ex) + '</div>';
            } else if (Array.isArray(ex)) {
                html += _renderLegacyFieldArray(ex);
            }

            if (data.demo_mode) {
                html += '<div class="alert alert-warning mt-2 mb-0 py-1 px-2" style="font-size:0.8rem;">'
                    + '<i class="fas fa-info-circle me-1"></i>Running in demo mode (model not loaded)';
                if (data.available_models && data.available_models.length) {
                    var names = data.available_models.map(function(m) { return m.label; }).join(', ');
                    html += '. Available: ' + escapeHtml(names);
                }
                html += '</div>';
            }

            html += buildFeedbackBar(data.interaction_id, 'explain');
            outputDiv.innerHTML = html;
        }
    });
}

/* ── Explain sub-renderers ───────────────────────────────────── */

function _renderExplainHeader(ex) {
    if (!ex.message_type || !ex.message_type.name) return '';
    const mt = ex.message_type;
    const catColors = {
        'session': 'secondary', 'pre-trade': 'primary', 'trade': 'success',
        'post-trade': 'info', 'market-data': 'warning', 'reference-data': 'dark',
        'business': 'danger', 'network': 'secondary'
    };
    const catBadge = catColors[mt.category] || 'secondary';
    let h = '<div class="alert alert-primary alert-custom mb-3 d-flex align-items-start">';
    h += '<div style="font-size:2rem; margin-right:12px;">📨</div><div>';
    h += '<h5 class="mb-1">' + escapeHtml(mt.name);
    h += ' <span class="badge bg-' + catBadge + ' ms-2" style="font-size:0.7rem;">' + escapeHtml((mt.category || '').replace('-', ' ')) + '</span>';
    h += ' <span class="badge bg-outline-secondary ms-1" style="font-size:0.65rem; border:1px solid #888; background:transparent; color:#555;">MsgType=' + escapeHtml(mt.code || '') + '</span>';
    h += '</h5>';
    if (mt.description) h += '<p class="mb-0" style="font-size:0.9rem; color:#555;">' + escapeHtml(mt.description) + '</p>';
    h += '</div></div>';
    return h;
}

function _renderExplainSummary(ex) {
    if (!ex.summary) return '';
    const summaryHtml = escapeHtml(ex.summary).replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    const paragraphs = summaryHtml.split(/\n\n+/).map(p => '<p style="margin-bottom:0.5rem; line-height:1.7;">' + p + '</p>').join('');
    let h = '<div class="card mb-3 border-0 shadow-sm"><div class="card-body">';
    h += '<h6 class="card-title text-muted mb-2"><i class="fas fa-comment-dots me-2"></i>What This Message Does';
    h += ' <span class="badge bg-light text-secondary border" style="font-size:0.6rem; font-weight:400;">source: knowledge base</span></h6>';
    h += paragraphs + '</div></div>';
    return h;
}

function _renderModelInsight(ex) {
    if (!ex.model_insight) return '';
    var mi = ex.model_insight;
    if (mi.source !== 'model' && mi.source !== 'error') {
        // No model loaded — skip insight card
        return '';
    }
    if (!mi.nl_interpretation && !mi.msg_type_knowledge) return '';

    var isKB = mi.knowledge_source === 'fix_reference';
    var sourceBadge = isKB
        ? '<span class="badge" style="font-size:0.6rem; font-weight:400; background:#0d6efd; color:#fff;">source: knowledge base</span>'
        : '<span class="badge" style="font-size:0.6rem; font-weight:400; background:#7c3aed; color:#fff;">source: trained model</span>';

    // Version brain badge — show which checkpoint brain produced this insight
    var brainBadge = '';
    var versionsTrained = ex.versions_trained || [];
    if (versionsTrained.length > 0) {
        brainBadge = ' <span class="badge" style="font-size:0.6rem; font-weight:400; background:rgba(16,185,129,0.15); color:#10b981; border:1px solid #10b981;">brain: FIX ' + escapeHtml(versionsTrained.join(', ')) + '</span>';
    } else if (ex.fix_version) {
        brainBadge = ' <span class="badge" style="font-size:0.6rem; font-weight:400; background:rgba(16,185,129,0.15); color:#10b981; border:1px solid #10b981;">brain: FIX ' + escapeHtml(ex.fix_version) + '</span>';
    }

    var h = '<div class="card mb-3 border-0 shadow-sm" style="border-left:3px solid ' + (isKB ? '#0d6efd' : '#7c3aed') + ' !important;">';
    h += '<div class="card-body">';
    h += '<h6 class="card-title mb-2" style="color:' + (isKB ? '#0d6efd' : '#7c3aed') + ';"><i class="fas fa-brain me-2"></i>Model Insight ';
    h += sourceBadge + brainBadge + '</h6>';

    if (mi.msg_type_knowledge) {
        h += '<div class="mb-2" style="line-height:1.6;"><strong>Learned knowledge:</strong> ' + escapeHtml(mi.msg_type_knowledge) + '</div>';
    }
    if (mi.nl_interpretation) {
        h += '<div class="mb-1" style="line-height:1.6;"><strong>Model interprets this as:</strong> <em>&ldquo;' + escapeHtml(mi.nl_interpretation) + '&rdquo;</em></div>';
    }
    if (mi.model_generated_fix) {
        h += '<div style="font-size:0.8rem; color:#666; margin-top:4px;">';
        h += '<i class="fas fa-check-circle text-success me-1"></i>Model verified: generated a matching FIX message from its interpretation</div>';
    }
    h += '</div></div>';
    return h;
}

function _renderFieldBreakdown(ex) {
    if (!ex.fields || !ex.fields.length) return '';

    const keyTags = ['35', '49', '56', '55', '54', '38', '44', '40', '59', '39', '150', '11', '37'];

    let h = '<div class="card border-0 shadow-sm"><div class="card-body p-2">';
    h += '<h6 class="card-title px-2 pt-1 text-muted"><i class="fas fa-list-alt me-2"></i>Field-by-Field Breakdown (' + ex.fields.length + ' fields)';
    h += ' <span class="badge bg-light text-secondary border" style="font-size:0.6rem; font-weight:400;">source: knowledge base</span></h6>';
    h += '<div style="max-height:500px; overflow-y:auto;">';
    h += '<table class="table table-sm mb-0 field-table" style="table-layout:fixed;">';
    h += '<thead class="table-light" style="position:sticky;top:0;z-index:1;"><tr>';
    h += '<th style="width:55px">Tag</th><th style="width:140px">Field</th><th style="width:100px">Value</th><th>Explanation</th>';
    h += '</tr></thead><tbody>';

    ex.fields.forEach((f, idx) => {
        const hasExtra = f.description || (f.possible_values && Object.keys(f.possible_values).length);
        const rowId = 'field-detail-' + f.tag + '-' + idx;
        const isKey = keyTags.includes(String(f.tag));
        const rowStyle = isKey ? ' style="cursor:pointer; background:#f8fbff;"' : (hasExtra ? ' style="cursor:pointer;"' : '');
        const clickAttr = hasExtra ? ' onclick="toggleFieldDetail(\'' + rowId + '\')"' : '';

        let valueCell = '<code>' + escapeHtml(String(f.value)) + '</code>';
        if (f.value_meaning) {
            valueCell += ' <span class="badge bg-light text-dark border" style="font-size:0.7rem;">' + escapeHtml(f.value_meaning) + '</span>';
        }

        const explainText = f.explanation || f.description || '';

        h += '<tr' + rowStyle + clickAttr + '>';
        h += '<td><strong>' + escapeHtml(String(f.tag)) + '</strong>';
        if (hasExtra) h += ' <i class="fas fa-chevron-down" style="font-size:0.5rem;color:#aaa;"></i>';
        h += '</td>';
        h += '<td style="word-break:break-word;">' + escapeHtml(f.name || '');
        if (f.type) h += ' <small class="text-muted d-block" style="font-size:0.7rem;">' + escapeHtml(f.type) + '</small>';
        h += '</td>';
        h += '<td>' + valueCell + '</td>';
        h += '<td style="font-size:0.85rem; color:#333;">' + escapeHtml(explainText) + '</td>';
        h += '</tr>';

        if (hasExtra) {
            h += '<tr id="' + rowId + '" style="display:none;">';
            h += '<td colspan="4" class="px-3 py-2" style="background:#f5f7fa; border-left:3px solid #4a90d9;">';
            if (f.description) {
                h += '<div style="font-size:0.85rem; margin-bottom:6px;"><strong>FIX Spec:</strong> ' + escapeHtml(f.description) + '</div>';
            }
            if (f.possible_values && Object.keys(f.possible_values).length) {
                h += '<div style="font-size:0.82rem;"><strong>All Possible Values:</strong></div>';
                h += '<div style="display:flex; flex-wrap:wrap; gap:4px; margin-top:4px;">';
                Object.entries(f.possible_values).forEach(([k, v]) => {
                    const isCurrent = String(k) === String(f.value);
                    const cls = isCurrent ? 'bg-success text-white' : 'bg-light text-dark border';
                    h += '<span class="badge ' + cls + '" style="font-size:0.75rem; font-weight:' + (isCurrent ? '600' : '400') + ';">';
                    h += escapeHtml(k) + ' = ' + escapeHtml(v);
                    if (isCurrent) h += ' ✓';
                    h += '</span>';
                });
                h += '</div>';
            }
            h += '</td></tr>';
        }
    });

    h += '</tbody></table></div></div></div>';
    return h;
}

function _renderLegacyFieldArray(ex) {
    let h = '<div style="max-height:400px; overflow-y:auto;">';
    h += '<table class="table table-sm field-table">';
    h += '<thead><tr><th>Tag</th><th>Field Name</th><th>Value</th></tr></thead><tbody>';
    ex.forEach(field => {
        h += '<tr>';
        h += '<td><strong>' + escapeHtml(field.tag) + '</strong></td>';
        h += '<td>' + escapeHtml(field.name) + '</td>';
        h += '<td><code>' + escapeHtml(field.value) + '</code></td>';
        h += '</tr>';
    });
    h += '</tbody></table></div>';
    return h;
}

// ── Complete FIX Message ────────────────────────────────────────

async function complete() {
    await _callEndpoint({
        inputId: 'completeInput',
        outputId: 'completeOutput',
        loadingId: 'completeLoading',
        emptyMsg: 'Please enter a partial FIX message',
        toast: '🎯 Message completed!',
        apiFn: function(input) {
            return authFetch(API_BASE + '/complete', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ partial: input })
            });
        },
        renderFn: function(data, outputDiv) {
            var html = '<div class="alert alert-success alert-custom">';
            if (data.demo_mode) {
                html += '<div class="badge bg-warning text-dark mb-3"><i class="fas fa-flask me-1"></i> Demo Mode</div><br>';
                if (data.available_models && data.available_models.length) {
                    var names = data.available_models.map(function(m) { return m.label; }).join(', ');
                    html += '<small class="text-muted">Available models: ' + escapeHtml(names) + '</small><br>';
                }
            }
            html += '<strong><i class="fas fa-check-circle me-2"></i>Message Completed Successfully!</strong></div>';
            html += '<div class="output-box">' + escapeHtml(data.completed) + '</div>';
            html += '<button class="btn btn-copy" onclick="copyToClipboard(\'' + escapeHtml(data.completed).replace(/'/g, "\\'") + '\')"><i class="fas fa-copy"></i> Copy to Clipboard</button>';
            html += _renderExplainHeader(data);
            html += _renderModelInsight(data);
            html += buildFeedbackBar(data.interaction_id, 'complete');
            outputDiv.innerHTML = html;
        }
    });
}
