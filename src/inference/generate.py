"""
Module: src.inference.generate
===============================

Core inference engine for FixProtoGPT.

Provides :class:`FixProtoGPTInference` — the primary entry-point for
all inference operations: free-form generation, natural-language-to-FIX
conversion, message explanation, validation, completion, and an
interactive REPL.

Heavy helpers are delegated to sibling modules:
    - :mod:`src.inference.explainer`   — field / message explanation builders
    - :mod:`src.inference.enrichment`  — FIX header/trailer enrichment

Coding Standards
----------------
- PEP 8  : Python Style Guide — naming, spacing, line length ≤ 120
- PEP 257 : Docstring Conventions — Google-style docstrings
- PEP 484 : Type Hints
- Google Python Style Guide

Author : FixProtoGPT Team
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.core.transformer import ModelConfig, create_model
from src.core.tokenizer import FixProtocolTokenizer
from src.inference.enrichment import enrich_fix_message
from src.inference.explainer import build_field_explanation, build_explain_summary
from src.inference.beam_search import beam_search_generate, score_fix_validity
from src.utils import paths
from src.utils.fix_enrichment import enrich_parsed_fields, extract_msg_type_info


class FixProtoGPTInference:
    """Inference engine for the FixProtoGPT model.

    Attributes:
        device:     PyTorch device string.
        tokenizer:  Loaded :class:`FixProtocolTokenizer`.
        model:      Loaded :class:`FixProtoGPT` in eval mode.
    """

    # ── Lifecycle ─────────────────────────────────────────────────

    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        device: str = "",  # auto-detected if empty
        fix_version: Optional[str] = None,
    ) -> None:
        """Initialise the inference engine.

        Args:
            model_path:     Path to a model checkpoint (``.pt``).
            tokenizer_path: Path to a tokenizer directory.
            device:         Compute device (``"cpu"``, ``"cuda"``, ``"mps"``).
                            Auto-detected if empty.
            fix_version:    FIX version key (e.g. ``"5.0SP2"``).
        """
        if not device:
            from src.utils.device import detect_device
            device = detect_device()
        self.device = device
        self.fix_version: Optional[str] = fix_version

        # Resolve version metadata for enrichment
        self._begin_string = "FIXT.1.1"
        self._appl_ver_id: Optional[str] = "9"
        if fix_version:
            try:
                from src.core.version_registry import get_version_info

                info = get_version_info(fix_version)
                if info:
                    self._begin_string = info.begin_string
                    self._appl_ver_id = info.appl_ver_id
            except Exception:
                pass

        # Load checkpoint metadata (versions trained, step, etc.)
        self._checkpoint_meta: Dict[str, Any] = {}
        ckpt_meta_path = Path(model_path).parent / "checkpoint_meta.json"
        if ckpt_meta_path.exists():
            try:
                import json as _json
                with open(ckpt_meta_path) as _f:
                    self._checkpoint_meta = _json.load(_f)
            except Exception:
                pass

        # Load tokenizer
        print("Loading tokenizer...")
        self.tokenizer = FixProtocolTokenizer()
        self.tokenizer.load(tokenizer_path)

        # Load model
        print("Loading model...")
        # Make TrainConfig available for safe unpickling
        _caller_main = sys.modules.get("__main__")
        try:
            from src.training.train import TrainConfig

            if _caller_main and not hasattr(_caller_main, "TrainConfig"):
                _caller_main.TrainConfig = TrainConfig
        except ImportError:
            TrainConfig = None  # type: ignore[assignment]

        # Use weights_only=True with safe globals for TrainConfig
        _safe_globals = [TrainConfig] if TrainConfig is not None else []
        try:
            checkpoint = torch.load(
                model_path, map_location=device, weights_only=True,
            )
        except Exception:
            # Fallback for legacy checkpoints that contain non-tensor objects
            import torch.serialization
            if _safe_globals:
                with torch.serialization.safe_globals(_safe_globals):
                    checkpoint = torch.load(
                        model_path, map_location=device, weights_only=True,
                    )
            else:
                checkpoint = torch.load(
                    model_path, map_location=device, weights_only=False,
                )
        model_config = self._load_model_config(model_path)
        self.model = create_model(model_config)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(device)
        self.model.eval()
        print(f"Model loaded on {device}")

    # ── Config loading ────────────────────────────────────────────

    @staticmethod
    def _load_model_config(model_path: str) -> ModelConfig:
        """Load :class:`ModelConfig` from bundled or project-level YAML.

        Args:
            model_path: Path to the checkpoint file.

        Returns:
            Populated :class:`ModelConfig` instance.
        """
        checkpoint_dir = Path(model_path).parent
        for candidate in [
            checkpoint_dir / "config" / "model_config.yaml",
            Path("config") / "model_config.yaml",
        ]:
            if candidate.exists():
                return ModelConfig.from_yaml(candidate)
        return ModelConfig()

    # ── Generation ────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        num_samples: int = 1,
    ) -> List[str]:
        """Generate text from a prompt.

        Args:
            prompt:         Input text prompt.
            max_new_tokens: Maximum number of new tokens.
            temperature:    Sampling temperature.
            top_k:          Top-k filtering parameter.
            top_p:          Nucleus sampling threshold.
            num_samples:    Number of independent samples.

        Returns:
            List of generated text strings.
        """
        input_ids = self.tokenizer.encode(
            prompt, add_special_tokens=True, for_generation=True
        )
        prompt_len = len(input_ids)
        input_ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        eos_id = self.tokenizer.eos_token_id
        eom_id = self.tokenizer.special_tokens.get("<|eom|>")

        outputs: List[str] = []
        for _ in range(num_samples):
            generated = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                eos_token_id=eos_id,
            )
            gen_ids = generated[0].tolist()
            if eom_id is not None:
                for idx in range(prompt_len, len(gen_ids)):
                    if gen_ids[idx] == eom_id:
                        gen_ids = gen_ids[: idx + 1]
                        break
            outputs.append(self.tokenizer.decode(gen_ids, skip_special_tokens=True))
        return outputs

    # ── NL → FIX ──────────────────────────────────────────────────

    def natural_language_to_fix(self, nl_prompt: str, **kwargs: Any) -> str:
        """Convert a natural-language instruction to a FIX message.

        Args:
            nl_prompt: Natural language (e.g. ``"Buy 100 shares of AAPL"``).
            **kwargs:  Extra generation parameters.

        Returns:
            Enriched FIX message string.
        """
        nl_ids = self.tokenizer.encode(
            nl_prompt, add_special_tokens=True, for_generation=True
        )
        fix_token_id = self.tokenizer.special_tokens.get("<|fix|>")
        if fix_token_id is not None:
            nl_ids.append(fix_token_id)

        prompt_len = len(nl_ids)
        input_ids = torch.tensor([nl_ids], dtype=torch.long, device=self.device)

        eos_id = self.tokenizer.eos_token_id
        eom_id = self.tokenizer.special_tokens.get("<|eom|>")

        gen_kwargs = {k: v for k, v in kwargs.items() if k != "num_samples"}
        generated = self.model.generate(
            input_ids,
            max_new_tokens=gen_kwargs.pop("max_new_tokens", 256),
            temperature=gen_kwargs.pop("temperature", 0.8),
            top_k=gen_kwargs.pop("top_k", 50),
            top_p=gen_kwargs.pop("top_p", 0.95),
            eos_token_id=eos_id,
        )

        gen_ids = generated[0].tolist()
        if eom_id is not None:
            for idx in range(prompt_len, len(gen_ids)):
                if gen_ids[idx] == eom_id:
                    gen_ids = gen_ids[: idx + 1]
                    break

        fix_output = self.tokenizer.decode(
            gen_ids[prompt_len:], skip_special_tokens=True
        )
        return enrich_fix_message(
            fix_output,
            begin_string=self._begin_string,
            appl_ver_id=self._appl_ver_id,
        )

    # ── Beam Search Generation ────────────────────────────────────

    @torch.no_grad()
    def generate_beam_search(
        self,
        prompt: str,
        beam_width: int = 4,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        length_penalty: float = 1.0,
        fix_validity_weight: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """Generate text using beam search with FIX-validity re-ranking.

        Produces *beam_width* candidate outputs and ranks them by a
        weighted combination of model confidence and FIX structural
        validity.

        Args:
            prompt:              Input text prompt.
            beam_width:          Number of beams.
            max_new_tokens:      Maximum tokens to generate.
            temperature:         Softmax temperature.
            length_penalty:      Length normalisation exponent.
            fix_validity_weight: Weight given to FIX validity score
                                  (0.0 = pure model score, 1.0 = pure validity).

        Returns:
            List of dicts with ``text``, ``log_prob``, ``fix_score`` keys,
            sorted best-first.
        """
        input_ids = self.tokenizer.encode(
            prompt, add_special_tokens=True, for_generation=True,
        )
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        results = beam_search_generate(
            model=self.model,
            input_ids=input_tensor,
            beam_width=beam_width,
            max_new_tokens=max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
            length_penalty=length_penalty,
            fix_validity_weight=fix_validity_weight,
            temperature=temperature,
            tokenizer=self.tokenizer,
        )

        outputs: List[Dict[str, Any]] = []
        for token_ids, log_prob, fix_score in results:
            text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            outputs.append({
                "text": text,
                "log_prob": round(log_prob, 4),
                "fix_score": round(fix_score, 4),
            })

        return outputs

    # ── Completion (original) ─────────────────────────────────────

    def complete_fix_message(self, partial_fix: str, **kwargs: Any) -> str:
        """Complete a partial FIX message and enrich it.

        Args:
            partial_fix: Partial FIX message.
            **kwargs:    Extra generation parameters.

        Returns:
            Completed and enriched FIX message.
        """
        outputs = self.generate(partial_fix, num_samples=1, **kwargs)
        return enrich_fix_message(
            outputs[0],
            begin_string=self._begin_string,
            appl_ver_id=self._appl_ver_id,
        )

    # ── Validation ────────────────────────────────────────────────

    def validate_fix_message(self, fix_message: str) -> Dict[str, Any]:
        """Perform deep structural and semantic validation of a FIX message.

        Goes beyond basic tag-presence checks:
            - Per-message-type required field validation
            - Value domain enforcement (Side, OrdType, ExecType, etc.)
            - Checksum verification (Tag 10)
            - BodyLength verification (Tag 9)
            - Field ordering heuristics

        Args:
            fix_message: FIX message string to validate.

        Returns:
            Dict with ``valid``, ``fields``, ``missing_required_fields``,
            ``num_fields``, ``warnings``, ``errors``, ``checksum_valid``,
            ``fix_score`` keys.
        """
        fields = self.tokenizer.parse_fix_message(fix_message)
        present = {f["tag"] for f in fields}
        tag_values: Dict[str, str] = {f["tag"]: f["value"] for f in fields}

        errors: List[str] = []
        warnings: List[str] = []

        # ── 1. Session-level required tags ────────────────────────
        session_required = {"8", "9", "35", "49", "56", "10"}
        missing_session = session_required - present
        if missing_session:
            errors.append(f"Missing session-level tags: {sorted(missing_session)}")

        # ── 2. Per-message-type required body tags ────────────────
        _msg_type_body: Dict[str, set] = {
            "D": {"11", "21", "55", "54", "38", "40", "60"},
            "8": {"37", "11", "17", "150", "39", "55", "54", "38"},
            "F": {"11", "41", "55", "54", "38", "60"},
            "G": {"11", "41", "55", "54", "38", "40", "60"},
            "V": {"262", "263", "264"},
            "A": {"98", "108"},
        }
        msg_type = tag_values.get("35", "")
        body_required = _msg_type_body.get(msg_type, set())
        missing_body = body_required - present
        if missing_body:
            errors.append(
                f"MsgType={msg_type}: missing body tags {sorted(missing_body)}"
            )

        # ── 3. Value domain enforcement ───────────────────────────
        _value_domains: Dict[str, set] = {
            "54": {"1", "2", "3", "4", "5", "6", "7", "8", "9"},  # Side
            "40": {"1", "2", "3", "4", "6", "7", "8", "9", "D", "E", "J", "K", "P"},  # OrdType
            "39": {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E"},  # OrdStatus
            "150": {"0", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "I"},  # ExecType
            "59": {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"},  # TimeInForce
            "21": {"1", "2", "3"},  # HandlInst
            "98": {"0", "1", "2", "3", "4", "5", "6"},  # EncryptMethod
            "263": {"0", "1", "2"},  # SubscriptionRequestType
            "22": {"1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B"},  # SecurityIDSource
        }
        for tag, valid_values in _value_domains.items():
            if tag in tag_values and tag_values[tag] not in valid_values:
                warnings.append(
                    f"Tag {tag} has unexpected value '{tag_values[tag]}'"
                    f" (expected one of {sorted(valid_values)})"
                )

        # ── 4. Numeric field type checks ─────────────────────────
        _numeric_tags = {"9", "34", "38", "14", "151", "108", "264"}
        for tag in _numeric_tags:
            if tag in tag_values:
                val = tag_values[tag]
                if not val.replace(".", "").replace("-", "").isdigit():
                    warnings.append(f"Tag {tag} should be numeric, got '{val}'")

        _price_tags = {"44", "6", "31", "270"}
        for tag in _price_tags:
            if tag in tag_values:
                val = tag_values[tag]
                try:
                    float(val)
                except ValueError:
                    warnings.append(f"Tag {tag} should be a price, got '{val}'")

        # ── 5. Checksum verification (Tag 10) ────────────────────
        checksum_valid: Optional[bool] = None
        if "10" in tag_values:
            expected_cksum = tag_values["10"]
            # Calculate actual checksum over all bytes up to 10=
            delimiter = "\x01" if "\x01" in fix_message else "|"
            body_end_marker = f"{delimiter}10="
            body_end_idx = fix_message.find("10=")
            if body_end_idx >= 0:
                body = fix_message[:body_end_idx]
                # Replace | with SOH for checksum calculation
                body_soh = body.replace("|", "\x01") + "\x01"
                computed = sum(ord(c) for c in body_soh) % 256
                computed_str = f"{computed:03d}"
                checksum_valid = computed_str == expected_cksum
                if not checksum_valid:
                    warnings.append(
                        f"Checksum mismatch: expected {expected_cksum}, computed {computed_str}"
                    )

        # ── 6. BeginString check ─────────────────────────────────
        if "8" in tag_values:
            bs = tag_values["8"]
            if not (bs.startswith("FIX") or bs.startswith("FIXT")):
                errors.append(f"Invalid BeginString: '{bs}'")

        # ── 7. Field ordering heuristic ──────────────────────────
        if fields:
            first_tag = fields[0]["tag"]
            if first_tag != "8":
                warnings.append("Tag 8 (BeginString) should be the first field")
            if len(fields) >= 2 and fields[1]["tag"] != "9":
                warnings.append("Tag 9 (BodyLength) should be the second field")
            last_tag = fields[-1]["tag"]
            if last_tag != "10":
                warnings.append("Tag 10 (CheckSum) should be the last field")

        # ── Combined results ─────────────────────────────────────
        is_valid = len(errors) == 0

        # Also compute the score from beam_search module
        fix_score = score_fix_validity(fix_message)

        return {
            "valid": is_valid,
            "fields": fields,
            "missing_required_fields": sorted(
                (session_required | body_required) - present
            ),
            "num_fields": len(fields),
            "errors": errors,
            "warnings": warnings,
            "checksum_valid": checksum_valid,
            "fix_score": round(fix_score, 4),
        }

    # ── Explanation ───────────────────────────────────────────────

    def explain_fix_message(self, fix_message: str) -> Dict[str, Any]:
        """Generate a rich, conversational explanation of a FIX message.

        Combines knowledge-base lookups, template-based field explanations,
        and model-generated insight.

        Args:
            fix_message: Complete FIX message string.

        Returns:
            Dict with ``summary``, ``model_insight``, ``fields``,
            ``message_type`` keys.
        """
        from src.data.scraper import FIXProtocolScraper

        parsed = self.tokenizer.parse_fix_message(fix_message)
        enriched = enrich_parsed_fields(
            parsed, FIXProtocolScraper.FIELDS, FIXProtocolScraper.ENUMERATIONS, full=True,
        )
        msg_info = extract_msg_type_info(enriched, FIXProtocolScraper.MESSAGE_TYPES)

        # Enhance tag-35 explanation with message-type context
        msg_type_field = next((f for f in enriched if f["tag"] == "35"), None)
        if msg_type_field and msg_info:
            mt = FIXProtocolScraper.MESSAGE_TYPES.get(msg_info.get("code", ""), {})
            if mt:
                msg_type_field["explanation"] = (
                    f"This message is a {mt.get('name', msg_info['code'])} message — "
                    f"{mt.get('description', 'a FIX protocol operation').lower()}."
                )

        template_summary = build_explain_summary(enriched, msg_info)
        model_insight = self._generate_model_insight(fix_message, enriched, msg_info)

        return {
            "summary": template_summary,
            "model_insight": model_insight,
            "fields": enriched,
            "message_type": msg_info,
            "fix_version": self.fix_version,
            "versions_trained": self._checkpoint_meta.get("fix_versions_trained", []),
        }

    # ── Model insight (public) ───────────────────────────────────

    def get_model_insight(self, fix_message: str) -> Dict[str, Any]:
        """Generate model insight for any FIX message.

        Parses the message, extracts message-type metadata, and probes
        the trained model for learned knowledge about the message.

        Args:
            fix_message: A FIX message string (complete or partial).

        Returns:
            Dict with ``model_insight`` and ``message_type`` keys.
        """
        from src.data.scraper import FIXProtocolScraper

        parsed = self.tokenizer.parse_fix_message(fix_message)
        enriched = enrich_parsed_fields(
            parsed, FIXProtocolScraper.FIELDS, FIXProtocolScraper.ENUMERATIONS, full=False,
        )
        msg_info = extract_msg_type_info(enriched, FIXProtocolScraper.MESSAGE_TYPES)

        insight = self._generate_model_insight(fix_message, enriched, msg_info)
        return {
            "model_insight": insight,
            "message_type": msg_info,
            "fix_version": self.fix_version,
            "versions_trained": self._checkpoint_meta.get("fix_versions_trained", []),
            "checkpoint_step": self._checkpoint_meta.get("step"),
        }

    # ── Model insight (private) ───────────────────────────────────

    @torch.no_grad()
    def _generate_model_insight(
        self,
        fix_message: str,
        enriched: List[Dict],
        msg_info: Dict,
    ) -> Dict[str, Any]:
        """Use the trained model to generate contextual insight.

        Probes the model with prompts that mirror training-data patterns
        to extract learned knowledge.

        Args:
            fix_message: Original FIX message.
            enriched:    Per-field enriched dicts.
            msg_info:    Message-type metadata.

        Returns:
            Dict with ``nl_interpretation``, ``msg_type_knowledge``, and
            ``source`` keys.
        """
        result: Dict[str, Any] = {
            "nl_interpretation": "",
            "msg_type_knowledge": "",
            "source": "model",
        }

        try:
            if not hasattr(self, "model") or self.model is None:
                result["source"] = "unavailable"
                return result

            # 1. NL interpretation — build a concise field summary
            key_fields: Dict[str, str] = {}
            for f in enriched:
                if f["tag"] in ("55", "54", "38", "44", "40", "59", "39", "150"):
                    vm = f.get("value_meaning")
                    key_fields[f["name"]] = vm if vm else f["value"]

            symbol = key_fields.get("Symbol", "")
            side_val = key_fields.get("Side", "")
            qty = key_fields.get("OrderQty", "")

            # Only show "Buy X shares of Y" when fields look realistic
            _placeholder_syms = {"SYMBOL", "UNKNOWN", ""}
            is_real_order = (
                symbol and side_val and qty
                and symbol.upper() not in _placeholder_syms
            )

            if is_real_order:
                side_word = side_val.lower() if side_val in ("Buy", "Sell") else side_val
                # Enrich symbol with company name for display
                try:
                    from src.data.symbol_resolver import lookup_symbol_name
                    _name = lookup_symbol_name(symbol)
                    sym_label = f"{symbol} ({_name})" if _name else symbol
                except Exception:
                    sym_label = symbol
                nl_prompt = f"{side_word.capitalize()} {qty} shares of {sym_label}"

                prompt_ids = self.tokenizer.encode(
                    nl_prompt, add_special_tokens=True, for_generation=True
                )
                input_t = torch.tensor(
                    [prompt_ids], dtype=torch.long, device=self.device
                )
                generated = self.model.generate(
                    input_t,
                    max_new_tokens=80,
                    temperature=0.3,
                    top_k=20,
                    top_p=0.9,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                gen_ids = generated[0].tolist()
                eom_id = self.tokenizer.special_tokens.get("<|eom|>")
                if eom_id is not None:
                    for idx in range(len(prompt_ids), len(gen_ids)):
                        if gen_ids[idx] == eom_id:
                            gen_ids = gen_ids[:idx]
                            break

                full_output = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
                result["nl_interpretation"] = nl_prompt

                gen_text = (
                    full_output[len(nl_prompt):].strip()
                    if len(full_output) > len(nl_prompt)
                    else ""
                )
                if gen_text and "35=" in gen_text:
                    fix_start = gen_text.index("35=")
                    prefix_search = gen_text[:fix_start]
                    if "8=FIX" in prefix_search:
                        fix_start = prefix_search.index("8=FIX")
                    result["model_generated_fix"] = True

            else:
                # Construct interpretation from message type + valid fields
                parts: List[str] = []
                msg_name = msg_info.get("name", "")
                if msg_name:
                    parts.append(msg_name)
                # Only include field details if they look like real values
                _placeholder = {"SYMBOL", "UNKNOWN", ""}
                if symbol and symbol.upper() not in _placeholder:
                    try:
                        from src.data.symbol_resolver import lookup_symbol_name
                        _name = lookup_symbol_name(symbol)
                        _sym = f"{symbol} ({_name})" if _name else symbol
                    except Exception:
                        _sym = symbol
                    parts.append(f"for {_sym}")
                if parts:
                    result["nl_interpretation"] = " ".join(parts)

            # 2. Message-type knowledge
            if msg_info and msg_info.get("code"):
                mt_prompt = f"FIX Message Type {msg_info['code']}: "
                mt_ids = self.tokenizer.encode(
                    mt_prompt, add_special_tokens=True, for_generation=True
                )
                input_t = torch.tensor(
                    [mt_ids], dtype=torch.long, device=self.device
                )
                generated = self.model.generate(
                    input_t,
                    max_new_tokens=80,
                    temperature=0.15,
                    top_k=5,
                    top_p=0.8,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                gen_ids = generated[0].tolist()
                eom_id = self.tokenizer.special_tokens.get("<|eom|>")
                if eom_id is not None:
                    for idx in range(len(mt_ids), len(gen_ids)):
                        if gen_ids[idx] == eom_id:
                            gen_ids = gen_ids[:idx]
                            break

                mt_output = self.tokenizer.decode(
                    gen_ids, skip_special_tokens=True
                ).strip()

                if mt_prompt.strip() in mt_output:
                    desc_part = mt_output[
                        mt_output.index(mt_prompt.strip()) + len(mt_prompt.strip()):
                    ].strip()
                else:
                    desc_part = mt_output

                if "8=FIX" in desc_part:
                    desc_part = desc_part[: desc_part.index("8=FIX")].strip()
                desc_part = desc_part.split("\n")[0].strip().rstrip("|")

                # Quality filter — relaxed to allow more model output
                kb_desc = msg_info.get("description", "").lower()
                desc_lower = desc_part.lower()
                if kb_desc:
                    kb_words = set(kb_desc.split())
                    model_words = set(desc_lower.split())
                    overlap = kb_words & model_words
                    overlap_ratio = len(overlap) / max(len(kb_words), 1)
                else:
                    overlap_ratio = 0.0

                alpha_chars = sum(1 for c in desc_part if c.isalpha())
                coherent = (
                    len(desc_part) > 5
                    and alpha_chars > len(desc_part) * 0.4
                    and (overlap_ratio >= 0.1 or not kb_desc)
                )
                if coherent:
                    result["msg_type_knowledge"] = (
                        f"FIX Message Type {msg_info['code']}: {desc_part}"
                    )

            # KB fallback — always provide message-type knowledge
            if not result["msg_type_knowledge"] and msg_info.get("description"):
                mt_name = msg_info.get("name", "Unknown")
                mt_code = msg_info.get("code", "")
                mt_desc = msg_info["description"]
                result["msg_type_knowledge"] = (
                    f"FIX {mt_name} (MsgType={mt_code}): {mt_desc}"
                )
                result["knowledge_source"] = "fix_reference"

        except Exception as e:
            result["error"] = str(e)
            result["source"] = "error"

        return result

    # ── Interactive REPL ──────────────────────────────────────────

    def interactive_mode(self) -> None:
        """Start an interactive chat loop (REPL)."""
        print("\n" + "=" * 60)
        print("FixProtoGPT Interactive Mode")
        print("=" * 60)
        print("\nCommands:")
        print("  /generate <prompt>  - Generate from prompt")
        print("  /nl2fix <text>      - Convert natural language to FIX")
        print("  /explain <message>  - Explain FIX message")
        print("  /validate <message> - Validate FIX message")
        print("  /quit               - Exit")
        print()

        while True:
            try:
                user_input = input("FixProtoGPT> ").strip()
                if not user_input:
                    continue
                if user_input.startswith("/quit"):
                    print("Goodbye!")
                    break
                elif user_input.startswith("/generate "):
                    outputs = self.generate(user_input[10:], num_samples=1)
                    print(f"\nGenerated:\n{outputs[0]}\n")
                elif user_input.startswith("/nl2fix "):
                    fix_msg = self.natural_language_to_fix(user_input[8:])
                    print(f"\nFIX Message:\n{fix_msg}\n")
                elif user_input.startswith("/explain "):
                    explanation = self.explain_fix_message(user_input[9:])
                    print(f"\n{explanation}\n")
                elif user_input.startswith("/validate "):
                    result = self.validate_fix_message(user_input[10:])
                    print("\nValidation Result:")
                    print(f"  Valid: {result['valid']}")
                    print(f"  Number of fields: {result['num_fields']}")
                    if result["missing_required_fields"]:
                        print(f"  Missing required fields: {result['missing_required_fields']}")
                    print()
                else:
                    outputs = self.generate(user_input, num_samples=1)
                    print(f"\n{outputs[0]}\n")
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}\n")


# ── CLI entry-point ───────────────────────────────────────────────────


def main() -> None:
    """CLI entry-point for stand-alone inference."""
    import argparse

    parser = argparse.ArgumentParser(description="FixProtoGPT Inference")
    parser.add_argument(
        "--model",
        type=str,
        default="model_store/checkpoints/best.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Path to tokenizer directory (default: from config)",
    )
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt for generation")
    parser.add_argument("--interactive", action="store_true", help="Start interactive mode")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k filtering")
    parser.add_argument("--top_p", type=float, default=0.95, help="Nucleus sampling threshold")
    parser.add_argument("--max_tokens", type=int, default=256, help="Maximum tokens to generate")

    args = parser.parse_args()

    if args.tokenizer is None:
        args.tokenizer = str(paths.tokenizer_dir())

    if not Path(args.model).exists():
        print(f"Error: Model checkpoint not found at {args.model}")
        print("\nTo use inference, you need to:")
        print("1. Prepare training data: python data/prepare_data.py")
        print("2. Train the model: python training/train.py")
        print("\nFor now, here's an example of what the system would generate:")
        example_nl = "Buy 100 shares of AAPL at market price"
        example_fix = (
            "8=FIXT.1.1|9=200|35=D|49=SENDER01|56=TARGET01|34=1|"
            "52=20260101-12:30:00.000|1128=9|11=ORD12345|21=1|"
            "55=AAPL|54=1|38=100|40=1|59=0|60=20260101-12:30:00.000|10=000|"
        )
        print(f"\nExample Natural Language: {example_nl}")
        print(f"Example FIX Message: {example_fix}")
        return

    engine = FixProtoGPTInference(args.model, args.tokenizer)

    if args.interactive:
        engine.interactive_mode()
    elif args.prompt:
        outputs = engine.generate(
            args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            num_samples=1,
        )
        print(f"\nGenerated:\n{outputs[0]}")
    else:
        print("\nDemo Mode - Generating examples...")
        examples = [
            "Create a new order for 100 shares of AAPL",
            "8=FIXT.1.1|9=200|35=D|",
        ]
        for example in examples:
            print(f"\nPrompt: {example}")
            outputs = engine.generate(example, num_samples=1)
            print(f"Output: {outputs[0]}\n")


if __name__ == "__main__":
    main()
