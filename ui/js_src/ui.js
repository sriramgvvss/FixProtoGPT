/**
 * FixProtoGPT — UI Helpers Module
 * Toast notifications, clipboard, HTML escaping, utility functions.
 * @module ui
 */

/**
 * Display an error alert inside a container element.
 * @param {HTMLElement} element - The container to render the error in
 * @param {string} message - Error message text
 */
function showError(element, message) {
    element.innerHTML =
        '<div class="alert alert-danger alert-custom">'
        + '<i class="fas fa-exclamation-circle"></i> '
        + escapeHtml(message) + '</div>';
}

/**
 * Escape HTML special characters to prevent XSS.
 * @param {string} text - Raw text
 * @returns {string} Escaped text safe for innerHTML
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * Toggle visibility of a field-detail row in the explain table.
 * @param {string} rowId - The DOM id of the detail row
 */
function toggleFieldDetail(rowId) {
    const row = document.getElementById(rowId);
    if (row) row.style.display = row.style.display === 'none' ? '' : 'none';
}

/**
 * Copy text to the system clipboard.
 * @param {string} text - Text to copy
 */
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        showToast('✅ Copied to clipboard!', 'success');
    }).catch(() => {
        showToast('❌ Failed to copy', 'error');
    });
}

/**
 * Show a transient toast notification.
 * @param {string} message - Toast message (may contain emoji / HTML entities)
 * @param {'success'|'error'} [type='success'] - Toast type
 */
function showToast(message, type = 'success', duration) {
    if (duration === undefined) duration = 3000;
    const existingToast = document.querySelector('.custom-toast');
    if (existingToast) existingToast.remove();

    var bgMap = {
        success: 'linear-gradient(135deg, #10b981 0%, #059669 100%)',
        danger: 'linear-gradient(135deg, #ef4444 0%, #dc2626 100%)',
        error: 'linear-gradient(135deg, #ef4444 0%, #dc2626 100%)',
        warning: 'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)',
    };

    const toast = document.createElement('div');
    toast.className = `custom-toast toast-${type}`;
    toast.innerHTML = message;
    var animDelay = ((duration - 400) / 1000).toFixed(1) + 's';
    toast.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 16px 24px;
        background: ${bgMap[type] || bgMap.success};
        color: white;
        border-radius: 12px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.3);
        z-index: 10000;
        font-weight: 600;
        max-width: 420px;
        animation: slideInRight 0.4s ease-out, slideOutRight 0.4s ease-out ${animDelay};
    `;

    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), duration);
}

/**
 * Build a feedback bar (thumbs up / down) HTML snippet.
 * @param {string} interactionId - Interaction UUID
 * @param {string} endpoint - Endpoint name (for tracking)
 * @returns {string} HTML string
 */
function buildFeedbackBar(interactionId, endpoint) {
    if (!interactionId) return '';
    return '<div class="feedback-bar mt-2 d-flex align-items-center gap-2" data-iid="' + escapeHtml(interactionId) + '">'
        + '<span style="font-size:0.8rem; color:#888;">Was this helpful?</span>'
        + '<button class="btn btn-sm btn-outline-success feedback-btn" '
        +   'onclick="sendFeedback(\'' + escapeHtml(interactionId) + '\',\'positive\',this)" title="Yes">'
        +   '<i class="fas fa-thumbs-up"></i></button>'
        + '<button class="btn btn-sm btn-outline-danger feedback-btn" '
        +   'onclick="sendFeedback(\'' + escapeHtml(interactionId) + '\',\'negative\',this)" title="No">'
        +   '<i class="fas fa-thumbs-down"></i></button>'
        + '<span class="feedback-status" style="font-size:0.75rem; color:#888;"></span>'
        + '</div>';
}

// ── Inject toast animation keyframes ────────────────────────────
if (!document.querySelector('#toast-animations')) {
    const style = document.createElement('style');
    style.id = 'toast-animations';
    style.textContent = `
        @keyframes slideInRight {
            from { transform: translateX(400px); opacity: 0; }
            to   { transform: translateX(0); opacity: 1; }
        }
        @keyframes slideOutRight {
            from { transform: translateX(0); opacity: 1; }
            to   { transform: translateX(400px); opacity: 0; }
        }
    `;
    document.head.appendChild(style);
}
