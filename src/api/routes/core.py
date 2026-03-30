"""
Module: src.api.routes.core
============================

Core API Blueprint for FixProtoGPT.

Endpoints: generate, nl2fix, explain, validate, complete, status,
           examples, versions, set-version.

Coding Standards
----------------
- PEP 8  : Python Style Guide
- PEP 257 : Docstring Conventions — Google-style docstrings
- PEP 484 : Type Hints
- Google Python Style Guide

Author : FixProtoGPT Team
"""

from flask import Blueprint, request, jsonify, session

import src.api.state as state
from src.api.routes.auth import login_required
from src.persistence.action_logger import log_user_action, log_debug
from src.core.version_registry import (
    list_installed, get_version_info, is_valid_version, default_version,
)
from src.utils import paths
from src.data.symbol_resolver import resolve_symbol

core_bp = Blueprint("core", __name__, url_prefix="/api")


# ── NL Demo Parser ────────────────────────────────────────────────

import re as _re
from typing import List as _List


def _split_multi_orders(nl_text: str) -> _List[str]:
    """Split a multi-order prompt into individual order strings.

    Recognises two separators:
    1. Newlines — each non-blank line is an order.
    2. The pattern ``… and <one|1> <buy|sell> …`` inside a single line.

    Returns:
        List of 1+ order strings (never empty — falls back to original).
    """
    # 1. Newline-separated
    lines = [ln.strip() for ln in nl_text.splitlines() if ln.strip()]
    if len(lines) > 1:
        return lines

    # 2. " and <one|1|> <buy|sell|short>" mid-sentence
    parts = _re.split(
        r'\s+and\s+(?=(?:one|1|\d+)?\s*(?:buy|sell|short)\b)',
        nl_text,
        flags=_re.IGNORECASE,
    )
    parts = [p.strip() for p in parts if p.strip()]
    return parts if parts else [nl_text]


def _parse_nl_for_demo(nl_text: str) -> dict:
    """Extract trading details from a natural-language prompt.

    Uses :func:`src.data.symbol_resolver.resolve_symbol` for company
    name → ticker resolution.  This provides a three-tier lookup:
    local cache → Twelve Data symbol store → hardcoded fallback dictionary.

    Returns:
        Dict with keys: symbol, qty, side, side_code, price, ord_type,
        ord_type_code, msg_type_code.
    """
    text = nl_text.lower()

    # ── Symbol: ticker, company name, crypto, or FX pair ────────
    symbol = "SYMBOL"

    # 0. Check for explicit FX pair notation: "EUR/USD", "eurusd"
    fx_pair_match = _re.search(
        r'\b([A-Za-z]{3})/?([A-Za-z]{3})\b', nl_text,
    )
    if fx_pair_match:
        candidate = fx_pair_match.group(0).lower().replace("/", "")
        # Try resolving as FX pair first
        from src.data.symbol_resolver import SymbolResolver, AssetClass
        if SymbolResolver.detect_asset_class(candidate) == AssetClass.FX:
            resolved = resolve_symbol(candidate)
            if resolved:
                symbol = resolved

    # 0b. Check for crypto keywords / names
    if symbol == "SYMBOL":
        _CRYPTO_KEYWORDS = {
            "bitcoin", "btc", "ethereum", "ether", "eth", "ripple",
            "xrp", "solana", "sol", "cardano", "ada", "dogecoin",
            "doge", "polkadot", "dot", "avalanche", "avax",
            "chainlink", "link", "litecoin", "ltc", "shiba",
            "shib", "polygon", "matic", "tron", "trx", "uniswap",
            "uni", "stellar", "xlm", "cosmos", "atom", "near",
            "algorand", "algo", "aptos", "apt", "arbitrum", "arb",
            "filecoin", "fil", "aave", "maker", "mkr", "pepe",
            "bonk", "render", "rndr", "injective", "inj", "sei",
            "celestia", "tia", "jupiter", "jup", "sui", "optimism",
        }
        for kw in _CRYPTO_KEYWORDS:
            if _re.search(r'\b' + kw + r'\b', text):
                resolved = resolve_symbol(kw)
                if resolved:
                    symbol = resolved
                    break

    # 1. Explicit ticker after keyword: "of AAPL", "for MSFT"
    if symbol == "SYMBOL":
        sym_match = _re.search(r'\b(?:of|for|symbol|ticker)\s+([A-Z]{1,5})\b', nl_text)
        if sym_match:
            symbol = sym_match.group(1)

    # 2. Extract company name and resolve via SymbolResolver
    #    (covers local cache and hardcoded fallback)
    if symbol == "SYMBOL":
        company_match = _re.search(
            r'\b(?:of|for|buy|sell|short|trade|order|purchase)\s+'
            r'([\w\s&\'-]+?)(?:\s+(?:shares?|stock|at|for|@|limit|market|stop)\b|$)',
            text,
        )
        if company_match:
            company_name = company_match.group(1).strip()
            # Avoid resolving pure numbers or very short tokens
            if company_name and not company_name.isdigit() and len(company_name) > 1:
                resolved = resolve_symbol(company_name)
                if resolved:
                    symbol = resolved

    # 3. If still unresolved, try each word as a company name
    if symbol == "SYMBOL":
        words = _re.findall(r'\b([a-z]{3,})\b', text)
        # Skip common non-company words
        _SKIP_WORDS = {
            "buy", "sell", "short", "trade", "order", "place",
            "market", "limit", "stop", "shares", "share", "stock",
            "quantity", "qty", "price", "the", "for", "and",
            "with", "please", "can", "you", "get", "want",
            "would", "like", "make", "put", "set", "new",
            "single", "create", "submit", "send", "execute",
            "fill", "cancel", "amend", "modify", "replace",
            "change", "status", "data", "subscribe", "quote",
            "one", "unit", "units",
        }
        for word in words:
            if word in _SKIP_WORDS:
                continue
            resolved = resolve_symbol(word)
            if resolved:
                symbol = resolved
                break

    # 4. Standalone uppercase ticker as last resort
    if symbol == "SYMBOL":
        sym_match = _re.search(r'\b([A-Z]{2,5})\b', nl_text)
        if sym_match:
            symbol = sym_match.group(1)

    # ── Quantity ─────────────────────────────────────────────────
    qty_match = _re.search(r'(\d+)\s*(?:shares?|units?|quantity)', text)
    if not qty_match:
        qty_match = _re.search(r'(?:for|qty|quantity)\s+(?:of\s+)?(?:=\s*)?(\d+)\s*(?:units?|shares?)?', text)
    if not qty_match:
        qty_match = _re.search(r'(?:qty|quantity)\s*(?:of\s*)?(?:=\s*)?(\d+)', text)
    if not qty_match:
        # Grab a number that appears right after buy/sell (before the symbol)
        qty_match = _re.search(r'\b(?:buy|sell|short)\s+(\d+)\b', text)
    if not qty_match:
        # Last resort: first standalone number NOT after 'at'/'@'/'price'
        for m in _re.finditer(r'\b(\d+(?:\.\d+)?)\b', text):
            # Skip if immediately preceded by price keyword
            prefix = text[:m.start()].rstrip()
            if prefix.endswith(('at', '@', 'price', '$')):
                continue
            if '.' not in m.group(1):  # quantities are integers
                qty_match = m
                break
    qty = qty_match.group(1) if qty_match else "100"

    # ── Side ─────────────────────────────────────────────────────
    if "sell" in text or "short" in text:
        side, side_code = "Sell", "2"
    else:
        side, side_code = "Buy", "1"

    # ── Order type / price ───────────────────────────────────────
    price_match = _re.search(r'(?:price|@|at)\s*\$?([\d.]+)', text)
    if "limit" in text:
        ord_type_code = "2"  # Limit
        price = price_match.group(1) if price_match else "100.00"
    elif "stop" in text:
        ord_type_code = "3"  # Stop
        price = price_match.group(1) if price_match else "100.00"
    elif "market" in text:
        ord_type_code = "1"  # Explicit market
        price = None
    elif price_match:
        # Price given ("at 210.00") without explicit type → infer Limit
        ord_type_code = "2"
        price = price_match.group(1)
    else:
        ord_type_code = "1"  # Market (default)
        price = None

    # ── Message type ─────────────────────────────────────────────
    if "cancel" in text:
        msg_type_code = "F"
    elif "market data" in text or "subscribe" in text or "quote" in text:
        msg_type_code = "V"
    elif "status" in text and "order" in text:
        msg_type_code = "H"
    elif any(w in text for w in ("amend", "modify", "replace", "change")):
        msg_type_code = "G"
    else:
        msg_type_code = "D"  # New Order Single

    return {
        "symbol": symbol,
        "qty": qty,
        "side": side,
        "side_code": side_code,
        "price": price,
        "ord_type_code": ord_type_code,
        "msg_type_code": msg_type_code,
    }


def _apply_nl_corrections(fix_message: str, nl_text: str) -> str:
    """Post-process model output so key fields reflect the NL request.

    The model sometimes hallucinates wrong symbols, quantities, sides,
    or prices.  This function extracts those details from the original
    natural-language prompt and patches the FIX string.

    If the model output is severely garbled (wrong message type or
    missing critical order tags), the function rebuilds the message
    from scratch using the parsed NL details and the model's header.

    Args:
        fix_message: Model-generated FIX string (``|``-delimited).
        nl_text:     Original natural-language prompt.

    Returns:
        Corrected FIX string.
    """
    p = _parse_nl_for_demo(nl_text)
    delim = "\x01" if "\x01" in fix_message else "|"

    # ── Parse existing fields ────────────────────────────────────
    raw_parts = [s for s in fix_message.rstrip(delim).split(delim) if "=" in s]
    tag_map: dict[str, str] = {}
    for part in raw_parts:
        tag, val = part.split("=", 1)
        tag = tag.strip()
        tag_map.setdefault(tag, val.strip())

    model_msg_type = tag_map.get("35", "")

    # ── Detect garbled output ────────────────────────────────────
    # If the model produced a completely wrong message type
    # (e.g. Logon "A" when user asked for Sell order = "D"),
    # or the output lacks tag 55 when a symbol was requested,
    # rebuild from scratch.
    expected_mt = p["msg_type_code"]
    has_symbol_tag = "55" in tag_map
    needs_rebuild = False

    if expected_mt == "D" and model_msg_type != "D":
        # ExecutionReport (8) is a response, not an order — rebuild as NewOrderSingle
        needs_rebuild = True
    elif expected_mt == "G" and model_msg_type not in ("D", "G"):
        needs_rebuild = True
    elif expected_mt == "F" and model_msg_type != "F":
        needs_rebuild = True
    elif expected_mt == "V" and model_msg_type != "V":
        needs_rebuild = True
    elif p["symbol"] != "SYMBOL" and not has_symbol_tag:
        needs_rebuild = True

    if needs_rebuild:
        from datetime import datetime, timezone
        import random

        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H:%M:%S.000")
        seq = str(random.randint(10_000, 99_999))
        begin = tag_map.get("8", "FIXT.1.1")
        sender = tag_map.get("49", f"SENDER{random.randint(1, 10)}")
        target = tag_map.get("56", f"TARGET{random.randint(1, 10)}")
        appl = tag_map.get("1128", "")
        appl_part = f"|1128={appl}" if appl else ""
        price_tag = f"|44={p['price']}" if p.get("price") else ""

        if expected_mt == "F":
            rebuilt = (
                f"8={begin}|35=F|49={sender}|56={target}"
                f"|11=CANCEL{seq}|41=ORD{seq}|55={p['symbol']}"
                f"|54={p['side_code']}|60={ts}{appl_part}|10=000|"
            )
        elif expected_mt == "V":
            rebuilt = (
                f"8={begin}|35=V|49={sender}|56={target}"
                f"|262=MDREQ{seq}|263=1|264=0|55={p['symbol']}{appl_part}|10=000|"
            )
        else:
            rebuilt = (
                f"8={begin}|35={expected_mt}|49={sender}|56={target}"
                f"|34={random.randint(1, 9999)}|52={ts}{appl_part}"
                f"|11=ORD{seq}|21=1|55={p['symbol']}|54={p['side_code']}"
                f"|38={p['qty']}|40={p['ord_type_code']}{price_tag}"
                f"|59=0|60={ts}|10=000|"
            )
        return rebuilt

    # ── Patch individual fields ──────────────────────────────────
    parts = fix_message.rstrip(delim).split(delim)
    idx_map: dict[str, int] = {}
    for idx, part in enumerate(parts):
        if "=" in part:
            tag = part.split("=", 1)[0].strip()
            idx_map.setdefault(tag, idx)

    corrections: dict[str, str] = {}

    # Symbol — always override when the parser found a real ticker
    if p["symbol"] != "SYMBOL":
        corrections["55"] = p["symbol"]

    # Quantity — override when the parser extracted a real number
    qty_match = _re.search(r'(\d+)', nl_text.lower())
    if qty_match:
        corrections["38"] = p["qty"]

    # Side — override when the prompt clearly says buy/sell
    lower = nl_text.lower()
    if "sell" in lower or "short" in lower or "buy" in lower:
        corrections["54"] = p["side_code"]

    # Price — override for limit/stop orders when a price was given;
    # remove price tag for market orders (model sometimes hallucinates it)
    if p["price"] is not None:
        corrections["44"] = p["price"]
    elif p["ord_type_code"] == "1" and "44" in idx_map:
        # Market order — remove hallucinated price tag
        del_idx = idx_map["44"]
        parts[del_idx] = None  # mark for removal

    # Apply corrections
    for tag, value in corrections.items():
        if tag in idx_map:
            parts[idx_map[tag]] = f"{tag}={value}"
        elif tag == "44":
            # Insert price before checksum if not present
            cs_idx = idx_map.get("10")
            if cs_idx is not None:
                parts.insert(cs_idx, f"44={value}")

    # Remove None entries (e.g. stripped price tag)
    parts = [p for p in parts if p is not None]

    return delim.join(parts) + delim


# ── Helpers ───────────────────────────────────────────────────────

def _session_version() -> str:
    """Return the FIX version selected by the current user session,
    falling back to the YAML default."""
    return session.get("fix_version", default_version())


def _session_engine():
    """Load (or return cached) engine for the session's FIX version."""
    return state.load_model(_session_version())


def _available_models() -> list:
    """Return list of versions that have a trained model checkpoint.

    Each entry is ``{"version": "4.4", "label": "FIX 4.4"}``.
    """
    return [
        {"version": v["version"], "label": v["label"]}
        for v in list_installed()
        if v.get("has_model")
    ]


def _model_unavailable_info(selected_version: str) -> dict:
    """Build a ``model_unavailable`` payload suggesting alternatives.

    Included in every demo-mode response so all surfaces can display
    a helpful "switch to …" prompt.
    """
    available = _available_models()
    suggestion = None
    if available:
        suggestion = f"Switch to {available[0]['label']} or another available model."
    return {
        "model_unavailable": True,
        "selected_version": selected_version,
        "available_models": available,
        "switch_suggestion": suggestion,
    }


def _apply_insight(resp: dict, insight_data: dict) -> None:
    """Copy model_insight, message_type, and brain info from insight_data into resp."""
    resp["model_insight"] = insight_data.get("model_insight", {})
    resp["message_type"] = insight_data.get("message_type", {})
    resp["versions_trained"] = insight_data.get("versions_trained", [])
    resp["fix_version"] = insight_data.get("fix_version") or resp.get("fix_version")


def _attach_insight(engine, resp: dict, messages: list, multi: bool, single_msg: str = "") -> None:
    """Attach model insight to *resp* — handles both multi-order and single-order cases.

    Args:
        engine:     Inference engine instance.
        resp:       Response dict to mutate.
        messages:   List of messages (used for multi-order per-order insight).
        multi:      Whether this is a multi-order response.
        single_msg: The single message output (used when ``multi`` is False).
    """
    if multi:
        order_insights = []
        for msg in messages:
            try:
                idata = engine.get_model_insight(msg)
                order_insights.append({
                    "model_insight": idata.get("model_insight", {}),
                    "message_type": idata.get("message_type", {}),
                    "versions_trained": idata.get("versions_trained", []),
                    "fix_version": idata.get("fix_version"),
                })
            except Exception:
                order_insights.append({})
        resp["order_insights"] = order_insights
        if order_insights and order_insights[0]:
            _apply_insight(resp, order_insights[0])
    else:
        try:
            insight_data = engine.get_model_insight(single_msg)
            _apply_insight(resp, insight_data)
        except Exception:
            pass


def _check_quota() -> dict | None:
    """Return a JSON-ready error dict if the user has exceeded their daily quota.

    Returns ``None`` when the request is allowed to proceed.
    """
    user_id = session.get("user_id")
    if not user_id:
        return None
    try:
        quota = state.user_manager.check_token_quota(user_id)
        if not quota["allowed"]:
            return {
                "error": "Daily token quota exceeded",
                "daily_used": quota["daily_used"],
                "daily_limit": quota["daily_limit"],
                "remaining": quota["remaining"],
            }
    except Exception:
        pass  # fail-open: if quota check errors, allow the request
    return None


def _count_tokens(text: str) -> int:
    """Count tokens for a text using the loaded tokenizer, or estimate."""
    engine = state.inference_engine
    if engine is not None and hasattr(engine, "tokenizer"):
        try:
            return len(engine.tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            pass
    # Rough estimate: ~4 chars per token (GPT-style)
    return max(1, len(text) // 4)


def _record_tokens(endpoint: str, input_text: str, output_text: str) -> dict:
    """Count tokens and record usage for the current user. Returns counts."""
    user_id = session.get("user_id")
    input_tokens = _count_tokens(input_text)
    output_tokens = _count_tokens(output_text)
    if user_id:
        try:
            state.user_manager.record_token_usage(
                user_id, endpoint, input_tokens, output_tokens
            )
        except Exception as exc:
            state.logger.warning("Token tracking error: %s", exc)
    return {"input_tokens": input_tokens, "output_tokens": output_tokens}


def _build_demo_fix(
    p: dict, begin: str, appl: str, *, include_body_len: bool = False,
) -> str:
    """Build a single demo FIX message from parsed NL fields."""
    import random as _rand
    oid = f"ORD{_rand.randint(10000, 99999)}"
    price_tag = f"|44={p['price']}" if p.get("price") else ""
    bl = "|9=178" if include_body_len else ""

    if p["msg_type_code"] == "F":  # Cancel
        return (
            f"8={begin}|35=F|49=SENDER|56=TARGET"
            f"|11=CANCEL001|41={oid}|55={p['symbol']}"
            f"|54={p['side_code']}{appl}|"
        )
    if p["msg_type_code"] == "V":  # Market Data Request
        return (
            f"8={begin}|35=V|49=SENDER|56=TARGET"
            f"|262=MDREQ001|263=1|264=0|55={p['symbol']}{appl}|"
        )
    if p["msg_type_code"] == "G":  # Amend
        return (
            f"8={begin}|35=G|49=SENDER|56=TARGET"
            f"|11={oid}|41=ORDER123|55={p['symbol']}"
            f"|54={p['side_code']}|38={p['qty']}"
            f"|40={p['ord_type_code']}{price_tag}{appl}|"
        )
    # Default: New Order Single (D)
    return (
        f"8={begin}{bl}|35={p['msg_type_code']}|49=SENDER|56=TARGET"
        f"|11={oid}|55={p['symbol']}|54={p['side_code']}"
        f"|38={p['qty']}|40={p['ord_type_code']}{price_tag}{appl}|"
    )


@core_bp.route("/generate", methods=["POST"])
@login_required
def generate():
    """Generate FIX message from prompt."""
    try:
        quota_err = _check_quota()
        if quota_err:
            return jsonify(quota_err), 429

        data = request.json
        prompt = data.get("prompt", "")
        temperature = data.get("temperature", 0.8)
        max_tokens = min(int(data.get("max_tokens", 256)), 512)

        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400

        log_debug("generate_request", detail={
            "prompt": prompt[:300], "temperature": temperature,
            "max_tokens": max_tokens,
        })

        ver = _session_version()
        engine = _session_engine()
        ver_info = get_version_info(ver)

        if engine is None:
            begin = ver_info.begin_string if ver_info else "FIXT.1.1"
            appl = f"|1128={ver_info.appl_ver_id}" if ver_info and ver_info.appl_ver_id else ""
            sub_orders = _split_multi_orders(prompt)
            msgs = [
                _build_demo_fix(
                    _parse_nl_for_demo(sub), begin, appl, include_body_len=True,
                )
                for sub in sub_orders
            ]
            demo_msg = "\n".join(msgs)
            multi = len(msgs) > 1
            resp = {
                "generated": demo_msg,
                "demo_mode": True,
                "multi_order": multi,
                "order_count": len(msgs),
                "message": f"Model not trained for FIX {ver}. This is a demo response.",
                "fix_version": ver,
            }
            resp.update(_model_unavailable_info(ver))
            return jsonify(resp)

        # ── Multi-order: split input and generate each sub-order ──
        sub_orders = _split_multi_orders(prompt)
        if len(sub_orders) > 1:
            all_outputs = []
            for sub in sub_orders:
                out = engine.generate(
                    sub, max_new_tokens=max_tokens, temperature=temperature, num_samples=1
                )
                all_outputs.append(out[0])
            generated = "\n".join(all_outputs)
            multi = True
        else:
            outputs = engine.generate(
                prompt, max_new_tokens=max_tokens, temperature=temperature, num_samples=1
            )
            generated = outputs[0]
            multi = False

        token_info = _record_tokens("generate", prompt, generated)
        resp = {
            "generated": generated,
            "demo_mode": False,
            "multi_order": multi,
            "order_count": len(sub_orders),
            "tokens": token_info,
            "fix_version": ver,
        }

        # Model insight — per-order for multi, single for solo
        _attach_insight(engine, resp, all_outputs if multi else [], multi, single_msg=generated)

        iid = state.interaction_log.log(
            "generate",
            {"prompt": prompt, "temperature": temperature, "max_tokens": max_tokens},
            resp,
            {"demo_mode": False},
            user_id=session.get("user_id"),
        )
        resp["interaction_id"] = iid
        log_user_action("generate", detail={
            "prompt": prompt[:200], "demo_mode": False,
            "multi_order": multi, "fix_version": ver,
        })
        return jsonify(resp)

    except Exception as e:
        state.logger.error("Generation error: %s", e)
        return jsonify({"error": str(e)}), 500


@core_bp.route("/nl2fix", methods=["POST"])
@login_required
def natural_language_to_fix():
    """Convert natural language to FIX message."""
    try:
        quota_err = _check_quota()
        if quota_err:
            return jsonify(quota_err), 429

        data = request.json
        nl_text = data.get("text", "")

        if not nl_text:
            return jsonify({"error": "Text is required"}), 400

        log_debug("nl2fix_request", detail={"text": nl_text[:300]})

        ver = _session_version()
        engine = _session_engine()
        ver_info = get_version_info(ver)
        begin = ver_info.begin_string if ver_info else "FIXT.1.1"
        appl = f"|1128={ver_info.appl_ver_id}" if ver_info and ver_info.appl_ver_id else ""

        if engine is None:
            sub_orders = _split_multi_orders(nl_text)
            msgs = [
                _build_demo_fix(_parse_nl_for_demo(sub), begin, appl)
                for sub in sub_orders
            ]
            demo_msg = "\n".join(msgs)
            multi = len(msgs) > 1

            return jsonify({
                "fix_message": demo_msg,
                "demo_mode": True,
                "multi_order": multi,
                "order_count": len(msgs),
                "message": f"Model not trained for FIX {ver}. Demo response.",
                "fix_version": ver,
            })

        # ── Multi-order: split input and generate each sub-order ──
        sub_orders = _split_multi_orders(nl_text)
        if len(sub_orders) > 1:
            msgs = []
            for sub in sub_orders:
                fm = engine.natural_language_to_fix(sub)
                fm = _apply_nl_corrections(fm, sub)
                msgs.append(fm)
            fix_message = "\n".join(msgs)
            multi = True
        else:
            fix_message = engine.natural_language_to_fix(nl_text)
            fix_message = _apply_nl_corrections(fix_message, nl_text)
            multi = False

        token_info = _record_tokens("nl2fix", nl_text, fix_message)
        resp = {
            "fix_message": fix_message,
            "demo_mode": False,
            "multi_order": multi,
            "order_count": len(sub_orders),
            "tokens": token_info,
            "fix_version": ver,
        }

        # Model insight — per-order for multi, single for solo
        _attach_insight(engine, resp, msgs if multi else [], multi, single_msg=fix_message)

        iid = state.interaction_log.log("nl2fix", {"text": nl_text}, resp, {"demo_mode": False}, user_id=session.get("user_id"))
        resp["interaction_id"] = iid
        log_user_action("nl2fix", detail={
            "text": nl_text[:200], "demo_mode": False,
            "multi_order": multi, "fix_version": ver,
        })
        return jsonify(resp)

    except Exception as e:
        state.logger.error("NL2FIX error: %s", e)
        return jsonify({"error": str(e)}), 500


@core_bp.route("/explain", methods=["POST"])
@login_required
def explain_fix():
    """Explain a FIX message with rich conversational detail."""
    try:
        data = request.json
        fix_message = data.get("message", "")

        if not fix_message:
            return jsonify({"error": "FIX message is required"}), 400

        engine = _session_engine()

        if engine is None:
            return _demo_explain(fix_message)

        explanation = engine.explain_fix_message(fix_message)

        import json as _json
        output_text = _json.dumps(explanation, default=str)
        token_info = _record_tokens("explain", fix_message, output_text)
        resp = {"explanation": explanation, "demo_mode": False, "tokens": token_info}
        iid = state.interaction_log.log("explain", {"message": fix_message}, resp, {"demo_mode": False}, user_id=session.get("user_id"))
        resp["interaction_id"] = iid
        log_user_action("explain", detail={"message": fix_message[:200]})
        return jsonify(resp)

    except Exception as e:
        state.logger.error("Explain error: %s", e)
        return jsonify({"error": str(e)}), 500


def _demo_explain(fix_message: str):
    """Return a rich explanation using the scraper knowledge base (no model)."""
    from src.data.scraper import FIXProtocolScraper
    from src.core.tokenizer import FixProtocolTokenizer
    from src.inference.explainer import build_explain_summary
    from src.utils.fix_enrichment import enrich_parsed_fields, extract_msg_type_info

    tokenizer = FixProtocolTokenizer()
    parsed = tokenizer.parse_fix_message(fix_message)

    enriched = enrich_parsed_fields(
        parsed, FIXProtocolScraper.FIELDS, FIXProtocolScraper.ENUMERATIONS, full=True,
    )
    msg_info = extract_msg_type_info(enriched, FIXProtocolScraper.MESSAGE_TYPES)
    summary = build_explain_summary(enriched, msg_info)

    resp = {
        "explanation": {
            "summary": summary,
            "model_insight": {
                "source": "unavailable",
                "nl_interpretation": "",
                "msg_type_knowledge": "",
            },
            "fields": enriched,
            "message_type": msg_info,
        },
        "demo_mode": True,
    }
    resp.update(_model_unavailable_info(_session_version()))
    return jsonify(resp)


@core_bp.route("/validate", methods=["POST"])
@login_required
def validate_fix():
    """Validate a FIX message."""
    try:
        data = request.json
        fix_message = data.get("message", "")

        if not fix_message:
            return jsonify({"error": "FIX message is required"}), 400

        engine = _session_engine()

        if engine is None:
            from src.core.tokenizer import FixProtocolTokenizer

            tokenizer = FixProtocolTokenizer()
            result = tokenizer.parse_fix_message(fix_message)
            required_fields = {"8", "9", "35", "49", "56", "10"}
            present_fields = {f["tag"] for f in result}
            missing = required_fields - present_fields

            resp = {
                "valid": len(missing) == 0,
                "fields": result,
                "missing_required_fields": list(missing),
                "num_fields": len(result),
                "demo_mode": True,
            }
            resp.update(_model_unavailable_info(_session_version()))
            return jsonify(resp)

        result = engine.validate_fix_message(fix_message)

        import json as _json
        output_text = _json.dumps(result, default=str)
        token_info = _record_tokens("validate", fix_message, output_text)
        resp = {**result, "demo_mode": False, "tokens": token_info}

        # Model insight
        try:
            insight_data = engine.get_model_insight(fix_message)
            _apply_insight(resp, insight_data)
        except Exception:
            pass

        iid = state.interaction_log.log("validate", {"message": fix_message}, resp, {"demo_mode": False}, user_id=session.get("user_id"))
        resp["interaction_id"] = iid
        log_user_action("validate", detail={"message": fix_message[:200]})
        return jsonify(resp)

    except Exception as e:
        state.logger.error("Validation error: %s", e)
        return jsonify({"error": str(e)}), 500


@core_bp.route("/complete", methods=["POST"])
@login_required
def complete_fix():
    """Complete a partial FIX message."""
    try:
        data = request.json
        partial_fix = data.get("partial", "")

        if not partial_fix:
            return jsonify({"error": "Partial FIX message is required"}), 400

        engine = _session_engine()

        if engine is None:
            completed = partial_fix + "54=1|38=100|40=2|44=150.00|59=0|10=123|"
            resp = {"completed": completed, "demo_mode": True}
            resp.update(_model_unavailable_info(_session_version()))
            return jsonify(resp)

        completed = engine.complete_fix_message(partial_fix)

        token_info = _record_tokens("complete", partial_fix, completed)
        resp = {"completed": completed, "demo_mode": False, "tokens": token_info}

        # Model insight
        try:
            insight_data = engine.get_model_insight(completed)
            _apply_insight(resp, insight_data)
        except Exception:
            pass

        iid = state.interaction_log.log("complete", {"partial": partial_fix}, resp, {"demo_mode": False}, user_id=session.get("user_id"))
        resp["interaction_id"] = iid
        log_user_action("complete", detail={"partial": partial_fix[:200]})
        return jsonify(resp)

    except Exception as e:
        state.logger.error("Completion error: %s", e)
        return jsonify({"error": str(e)}), 500


@core_bp.route("/status", methods=["GET"])
@login_required
def status():
    """Get system status including active FIX version and checkpoint info."""
    import json as _json

    ver = _session_version()
    _engine = state.get_engine(ver)

    model_loaded = _engine is not None
    model_path_exists = paths.best_model(ver).exists()
    interaction_stats = state.interaction_log.get_stats()

    # Include current user's token usage
    user_id = session.get("user_id")
    token_usage = {}
    if user_id:
        try:
            token_usage = state.user_manager.get_user_token_usage(user_id)
        except Exception:
            pass

    ver_info = get_version_info(ver)

    # ── Checkpoint / brain metadata ───────────────────────────────
    checkpoint_info = {}
    ckpt_dir = paths.checkpoint_dir(ver)
    meta_file = ckpt_dir / "checkpoint_meta.json"
    if meta_file.exists():
        try:
            with open(meta_file) as f:
                ckpt_meta = _json.load(f)
            checkpoint_info = {
                "step": ckpt_meta.get("step"),
                "epoch": ckpt_meta.get("epoch"),
                "best_val_loss": ckpt_meta.get("best_val_loss"),
                "fix_versions_trained": ckpt_meta.get("fix_versions_trained", []),
                "model_params": ckpt_meta.get("model_params"),
                "saved_at": ckpt_meta.get("saved_at"),
                "checkpoint_dir": str(ckpt_dir.name),
            }
        except Exception:
            pass

    # demo_mode is True when no engine is loaded — user gets template
    # responses regardless of whether the checkpoint file exists on disk.
    demo_mode = not model_loaded

    return jsonify({
        "model_loaded": model_loaded,
        "model_available": model_path_exists,
        "demo_mode": demo_mode,
        "engine_mode": "active" if model_loaded else "demo",
        "version": "1.0.0",
        "fix_version": ver,
        "fix_version_label": ver_info.label if ver_info else f"FIX {ver}",
        "interactions": interaction_stats,
        "token_usage": token_usage,
        "checkpoint_info": checkpoint_info,
    })


# ── Version Management ────────────────────────────────────────────

@core_bp.route("/versions", methods=["GET"])
@login_required
def list_versions():
    """Return all known FIX versions with their install/model status."""
    versions = list_installed()
    # Mark the session's selected version
    current = _session_version()
    for v in versions:
        v["selected"] = (v["version"] == current)
    return jsonify({"versions": versions, "current": current})


@core_bp.route("/version", methods=["POST"])
@login_required
def set_version():
    """Set the active FIX version for this user's session."""
    data = request.get_json(silent=True) or {}
    ver = data.get("version", "").strip()

    if not ver:
        return jsonify({"error": "version is required"}), 400

    if not is_valid_version(ver):
        return jsonify({"error": f"Unknown FIX version: {ver}"}), 400

    session["fix_version"] = ver

    info = get_version_info(ver)
    has_model = paths.best_model(ver).exists()

    # Trigger lazy loading of the engine (non-blocking in future requests)
    state.load_model(ver)

    log_user_action("set_version", detail={"version": ver, "has_model": has_model})
    resp = {
        "success": True,
        "version": ver,
        "label": info.label if info else ver,
        "has_model": has_model,
        "demo_mode": not has_model,
    }
    if not has_model:
        resp.update(_model_unavailable_info(ver))
    return jsonify(resp)


@core_bp.route("/examples", methods=["GET"])
@login_required
def get_examples():
    """Get example queries."""
    examples = [
        {
            "category": "Smart Prompts",
            "examples": [
                "Buy 100 shares of AAPL at market price",
                "Sell 50 shares of GOOGL at limit price $150",
                "Create a market data request for MSFT",
                "Cancel order ORDER123",
            ],
        },
        {
            "category": "FIX Messages",
            "examples": [
                "8=FIXT.1.1|9=178|35=D|55=AAPL|54=1|38=100|40=2|44=150.50|1128=9|",
                "8=FIXT.1.1|35=8|55=GOOGL|54=2|38=50|150=2|39=2|1128=9|",
                "8=FIXT.1.1|35=V|262=MDREQ001|263=1|55=MSFT|1128=9|",
            ],
        },
    ]
    return jsonify({"examples": examples})


# ── Symbol Resolution ─────────────────────────────────────────────

@core_bp.route("/symbols/resolve", methods=["POST"])
@login_required
def resolve_company_symbol():
    """Resolve a company name to its stock ticker symbol.

    Uses the two-tier lookup: cache → fallback dict.

    Request JSON: ``{"query": "google"}``

    Response JSON::

        {
            "ticker": "GOOGL",
            "query": "google",
            "source": "cache"
        }
    """
    data = request.get_json(silent=True) or {}
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "query is required"}), 400

    from src.data.symbol_resolver import get_resolver
    resolver = get_resolver()
    ticker = resolver.resolve(query)
    if ticker:
        return jsonify({
            "ticker": ticker,
            "query": query,
            "cache_size": resolver.cache_size,
        })
    return jsonify({
        "ticker": None,
        "query": query,
        "message": f"Could not resolve '{query}' to a ticker symbol",
    }), 404


@core_bp.route("/symbols/cache", methods=["GET"])
@login_required
def get_symbol_cache():
    """Return all cached company→ticker mappings."""
    from src.data.symbol_resolver import get_resolver
    resolver = get_resolver(use_api=False)
    entries = resolver.cached_entries
    return jsonify({
        "cache_size": len(entries),
        "entries": entries,
    })


@core_bp.route("/symbols/sync", methods=["POST"])
@login_required
def sync_symbols():
    """Trigger a Twelve Data symbol sync.

    Fetches all equities, forex, and crypto from the Twelve Data API
    and rebuilds the resolver cache.

    Request JSON (all optional)::

        {
            "stocks": true,
            "forex": true,
            "crypto": true,
            "etfs": false,
            "api_key": "optional_override"
        }

    Response JSON::

        {
            "status": "ok",
            "counts": {"stocks": 16559, "forex": 1459, "crypto": 2101}
        }
    """
    data = request.get_json(silent=True) or {}

    from src.data.twelve_data import get_symbol_store
    store = get_symbol_store(api_key=data.get("api_key"))

    try:
        counts = store.sync(
            stocks=data.get("stocks", True),
            forex=data.get("forex", True),
            crypto=data.get("crypto", True),
            etfs=data.get("etfs", False),
        )

        # Rebuild the resolver cache
        from src.data.symbol_resolver import get_resolver
        resolver = get_resolver(use_api=False)
        combined = store.build_combined_map()
        if combined:
            resolver._cache.put_many(combined)

        return jsonify({
            "status": "ok",
            "counts": counts,
            "cache_size": resolver.cache_size,
        })
    except Exception as exc:
        return jsonify({
            "status": "error",
            "message": str(exc),
        }), 500


@core_bp.route("/symbols/sync/status", methods=["GET"])
@login_required
def sync_status():
    """Return the status of the last Twelve Data sync.

    Response JSON::

        {
            "synced": true,
            "last_sync": "2026-02-17T02:00:00+00:00",
            "counts": {"stocks": 16559, "forex": 1459, "crypto": 2101}
        }
    """
    from src.data.twelve_data import get_symbol_store
    store = get_symbol_store()
    meta = store.load_sync_meta()
    return jsonify({
        "synced": store.is_synced(),
        **meta,
    })
