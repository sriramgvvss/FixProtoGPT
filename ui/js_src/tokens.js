/**
 * FixProtoGPT — Token Usage Panel Module
 * Displays per-user token consumption with admin all-users overview.
 * Depends on: api.js
 * @module tokens
 */

/**
 * Format a number with commas for display.
 * @param {number} n
 * @returns {string}
 */
function formatTokenCount(n) {
    if (n >= 1000000) return (n / 1000000).toFixed(1) + 'M';
    if (n >= 1000) return (n / 1000).toFixed(1) + 'K';
    return String(n);
}

/**
 * Update the token badge in the header from status data.
 * @param {object} usage - {total_tokens, input_tokens, output_tokens, request_count, by_endpoint}
 */
function updateTokenBadge(usage) {
    const badge = document.getElementById('tokenBadge');
    const countEl = document.getElementById('tokenCount');
    if (!badge || !countEl) return;

    const total = (usage && usage.total_tokens) ? usage.total_tokens : 0;
    badge.style.display = 'inline-block';
    countEl.textContent = formatTokenCount(total);

    // Cache usage data for the panel
    window._tokenUsageCache = usage || {};
}

/**
 * Fetch the latest token totals from the server and update the header badge.
 * Call this after any action that consumes tokens (generate, validate, etc.).
 */
async function refreshTokenBadge() {
    try {
        const resp = await authFetch('/auth/token-usage');
        const data = await resp.json();
        updateTokenBadge(data);
    } catch (_) { /* silent — badge will refresh on next status poll */ }
}

/**
 * Toggle the token usage panel visibility.
 */
function toggleTokenPanel() {
    const panel = document.getElementById('tokenPanel');
    if (!panel) return;

    if (panel.style.display === 'none') {
        panel.style.display = 'block';
        loadTokenPanel();
    } else {
        panel.style.display = 'none';
    }
}

/**
 * Populate the token panel with current user stats + admin table.
 */
async function loadTokenPanel() {
    // Current user stats
    try {
        const resp = await authFetch('/auth/token-usage');
        const data = await resp.json();

        document.getElementById('tpTotal').textContent = formatTokenCount(data.total_tokens || 0);
        document.getElementById('tpInput').textContent = formatTokenCount(data.input_tokens || 0);
        document.getElementById('tpOutput').textContent = formatTokenCount(data.output_tokens || 0);
        document.getElementById('tpRequests').textContent = formatTokenCount(data.request_count || 0);

        // Per-endpoint breakdown
        const epSection = document.getElementById('tpEndpoints');
        const epList = document.getElementById('tpEndpointList');
        if (data.by_endpoint && Object.keys(data.by_endpoint).length > 0) {
            epSection.style.display = 'block';
            epList.innerHTML = '';
            for (const [ep, stats] of Object.entries(data.by_endpoint)) {
                const total = (stats.input_tokens || 0) + (stats.output_tokens || 0);
                const row = document.createElement('div');
                row.className = 'token-endpoint-row';
                row.innerHTML =
                    '<span class="ep-name">' + ep + '</span>' +
                    '<span><span class="ep-tokens">' + formatTokenCount(total) + '</span>' +
                    ' <small style="color:rgba(255,255,255,0.4);">(' + (stats.requests || 0) + ' reqs)</small></span>';
                epList.appendChild(row);
            }
        } else {
            epSection.style.display = 'none';
        }
    } catch (e) {
        console.error('Token usage fetch error:', e);
    }

    // Admin: all users table
    try {
        const resp = await authFetch('/auth/admin/token-usage');
        if (resp.status === 403) {
            document.getElementById('tpAdminSection').style.display = 'none';
            return;
        }
        const data = await resp.json();
        const section = document.getElementById('tpAdminSection');
        const tbody = document.getElementById('tpAdminTable');

        if (data.users && data.users.length > 0) {
            section.style.display = 'block';
            tbody.innerHTML = '';
            for (const u of data.users) {
                const tr = document.createElement('tr');
                tr.style.borderColor = 'rgba(255,255,255,0.06)';
                tr.innerHTML =
                    '<td>' + u.username +
                        (u.role === 'admin' ? '<span class="admin-badge">admin</span>' : '') +
                    '</td>' +
                    '<td class="text-end" style="color:rgba(255,255,255,0.7);">' + formatTokenCount(u.input_tokens) + '</td>' +
                    '<td class="text-end" style="color:rgba(255,255,255,0.7);">' + formatTokenCount(u.output_tokens) + '</td>' +
                    '<td class="text-end" style="color:#f59e0b; font-weight:600;">' + formatTokenCount(u.total_tokens) + '</td>' +
                    '<td class="text-end" style="color:rgba(255,255,255,0.5);">' + u.request_count + '</td>';
                tbody.appendChild(tr);
            }
        } else {
            section.style.display = 'none';
        }
    } catch (e) {
        // Non-admin users will get 403, which is fine
        document.getElementById('tpAdminSection').style.display = 'none';
    }
}
