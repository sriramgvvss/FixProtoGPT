"""
Shared pytest fixtures and configuration for FixProtoGPT test suite.
"""

import sys
from pathlib import Path

import pytest
import torch

# Ensure the project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.tokenizer import FixProtocolTokenizer
from src.core.transformer import ModelConfig, FixProtoGPT
from src.utils.fix_converter import FixMessageConverter


# ── Sample FIX Messages ────────────────────────────────────────────

SAMPLE_NEW_ORDER = (
    "8=FIXT.1.1|9=200|35=D|49=SENDER1|56=TARGET1|34=100|"
    "52=20260101-12:30:00.000|1128=9|11=ORD123|21=1|55=AAPL|"
    "54=1|38=100|40=2|44=150.50|59=0|60=20260101-12:30:00.000|10=123|"
)

SAMPLE_EXEC_REPORT = (
    "8=FIXT.1.1|9=280|35=8|49=TARGET1|56=SENDER1|34=200|"
    "52=20260101-12:30:01.000|1128=9|37=EXEC456|11=ORD123|"
    "17=EXEC789|150=F|39=1|55=AAPL|54=1|38=100|14=50|151=50|"
    "6=150.25|60=20260101-12:30:01.000|10=456|"
)

SAMPLE_CANCEL_REQUEST = (
    "8=FIXT.1.1|9=150|35=F|49=SENDER1|56=TARGET1|34=300|"
    "52=20260101-12:31:00.000|1128=9|11=CANCEL789|41=ORD123|"
    "55=AAPL|54=1|38=100|60=20260101-12:31:00.000|10=789|"
)

SAMPLE_LOGON = (
    "8=FIXT.1.1|9=80|35=A|49=SENDER1|56=TARGET1|34=1|"
    "52=20260101-12:00:00.000|1128=9|98=0|108=30|10=100|"
)

SAMPLE_MD_REQUEST = (
    "8=FIXT.1.1|9=120|35=V|49=SENDER1|56=TARGET1|34=50|"
    "52=20260101-12:30:00.000|1128=9|262=MDREQ1|263=1|264=0|10=200|"
)


# ── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def tokenizer():
    """Create a fresh FixProtocolTokenizer with vocabulary built from samples."""
    tok = FixProtocolTokenizer(vocab_size=1024)
    sample_texts = [
        SAMPLE_NEW_ORDER,
        SAMPLE_EXEC_REPORT,
        SAMPLE_CANCEL_REQUEST,
        SAMPLE_LOGON,
        SAMPLE_MD_REQUEST,
        "Buy 100 shares of AAPL at market price",
        "Sell 50 shares of MSFT at limit price 410.00",
        "Cancel order ORD12345 for symbol GOOG",
    ]
    tok.build_vocab(sample_texts)
    return tok


@pytest.fixture
def small_model_config():
    """Return a small ModelConfig suitable for fast unit tests."""
    return ModelConfig(
        n_layers=2,
        n_heads=4,
        d_model=128,
        d_ff=256,
        vocab_size=1024,
        max_seq_len=64,
        dropout=0.0,
        attention_dropout=0.0,
        use_rotary=True,
        bias=False,
    )


@pytest.fixture
def small_model(small_model_config):
    """Create a small FixProtoGPT model for testing."""
    model = FixProtoGPT(small_model_config)
    model.eval()
    return model


@pytest.fixture
def converter(tokenizer):
    """Create a FixMessageConverter."""
    return FixMessageConverter(tokenizer=tokenizer)


@pytest.fixture
def sample_messages():
    """Return a dict of sample FIX messages by type."""
    return {
        "new_order": SAMPLE_NEW_ORDER,
        "exec_report": SAMPLE_EXEC_REPORT,
        "cancel_request": SAMPLE_CANCEL_REQUEST,
        "logon": SAMPLE_LOGON,
        "md_request": SAMPLE_MD_REQUEST,
    }


@pytest.fixture
def device():
    """Return CPU device for testing."""
    return torch.device("cpu")
