"""
Module: src.core.fix_reference
===============================

FIX Protocol 5.0 SP2 Reference Data — static spec validation data.

This module contains static reference data for the FIX 5.0 SP2
specification.  It is used for constrained decoding (validating
tag numbers during generation), training data bootstrapping, and
human-readable field/enum lookups in the explainer.  The trained
model generates FIX messages; this module provides the guardrails.

Coding Standards
----------------
- PEP 8  : Python Style Guide — naming, spacing, line length ≤ 120
- PEP 257 : Docstring Conventions — module / constant documentation
- PEP 484 : Type Hints
- Google Python Style Guide (docstring format)

Author : FixProtoGPT Team
"""

from __future__ import annotations

from typing import Any, Dict

# ─── FIX 5.0 SP2 Message Types ──────────────────────────────────────
# Maps MsgType (tag 35) code → metadata dict.
#   name        : human-readable message name
#   category    : workflow category (session | pre-trade | trade | …)
#   description : short functional description

MESSAGE_TYPES: Dict[str, Dict[str, str]] = {
    # Session (FIXT 1.1)
    "0": {"name": "Heartbeat", "category": "session", "description": "Sent when no data has been sent for HeartBtInt seconds"},
    "1": {"name": "TestRequest", "category": "session", "description": "Force a heartbeat from the opposing application"},
    "2": {"name": "ResendRequest", "category": "session", "description": "Request retransmission of messages"},
    "3": {"name": "Reject", "category": "session", "description": "Session-level rejection of a message"},
    "4": {"name": "SequenceReset", "category": "session", "description": "Reset expected sequence number"},
    "5": {"name": "Logout", "category": "session", "description": "Initiate or confirm session termination"},
    "A": {"name": "Logon", "category": "session", "description": "Authenticate and initiate a session"},

    # Pre-Trade
    "D": {"name": "NewOrderSingle", "category": "pre-trade", "description": "Submit a new single order"},
    "E": {"name": "NewOrderList", "category": "pre-trade", "description": "Submit a list of related orders"},
    "F": {"name": "OrderCancelRequest", "category": "pre-trade", "description": "Request cancellation of an order"},
    "G": {"name": "OrderCancelReplaceRequest", "category": "pre-trade", "description": "Request modification of an existing order"},
    "H": {"name": "OrderStatusRequest", "category": "pre-trade", "description": "Request status of an order"},
    "AB": {"name": "NewOrderMultileg", "category": "pre-trade", "description": "Submit a multi-leg order"},
    "AC": {"name": "MultilegOrderCancelReplace", "category": "pre-trade", "description": "Modify a multi-leg order"},
    "AF": {"name": "OrderMassStatusRequest", "category": "pre-trade", "description": "Request status of a group of orders"},
    "AQ": {"name": "OrderMassActionRequest", "category": "pre-trade", "description": "Request mass action on orders"},
    "CA": {"name": "NewOrderCross", "category": "pre-trade", "description": "Submit a cross order"},
    "CB": {"name": "CrossOrderCancelReplaceRequest", "category": "pre-trade", "description": "Modify a cross order"},
    "CC": {"name": "CrossOrderCancelRequest", "category": "pre-trade", "description": "Cancel a cross order"},
    "q": {"name": "OrderMassCancelRequest", "category": "pre-trade", "description": "Request mass cancellation of orders"},
    "r": {"name": "OrderMassCancelReport", "category": "pre-trade", "description": "Report of mass cancel action"},

    # Trade
    "8": {"name": "ExecutionReport", "category": "trade", "description": "Confirm receipt, rejection, or execution of an order"},
    "9": {"name": "OrderCancelReject", "category": "trade", "description": "Reject a cancel or cancel/replace request"},
    "Q": {"name": "DontKnowTrade", "category": "trade", "description": "Reject an execution or trade report"},

    # Post-Trade
    "AE": {"name": "TradeCaptureReport", "category": "post-trade", "description": "Report a trade between counterparties"},
    "AD": {"name": "TradeCaptureReportRequest", "category": "post-trade", "description": "Request trade capture reports"},
    "AQ": {"name": "TradeCaptureReportRequestAck", "category": "post-trade", "description": "Acknowledge trade capture report request"},
    "AZ": {"name": "TradeCaptureReportAck", "category": "post-trade", "description": "Acknowledge a trade capture report"},
    "J": {"name": "Allocation", "category": "post-trade", "description": "Convey allocation instructions"},
    "P": {"name": "AllocationAck", "category": "post-trade", "description": "Acknowledge an allocation"},
    "AS": {"name": "AllocationReport", "category": "post-trade", "description": "Report allocation details"},
    "AT": {"name": "AllocationReportAck", "category": "post-trade", "description": "Acknowledge an allocation report"},
    "AK": {"name": "Confirmation", "category": "post-trade", "description": "Confirm trade allocation to a client"},
    "AU": {"name": "ConfirmationAck", "category": "post-trade", "description": "Acknowledge confirmation"},
    "BH": {"name": "ConfirmationRequest", "category": "post-trade", "description": "Request a confirmation"},
    "T": {"name": "SettlementInstructions", "category": "post-trade", "description": "Convey settlement instructions"},
    "AV": {"name": "SettlementInstructionRequest", "category": "post-trade", "description": "Request settlement instructions"},
    "BN": {"name": "PositionReport", "category": "post-trade", "description": "Report a position"},
    "AM": {"name": "PositionMaintenanceRequest", "category": "post-trade", "description": "Request a position change"},
    "AN": {"name": "PositionMaintenanceReport", "category": "post-trade", "description": "Report result of position maintenance"},
    "AP": {"name": "CollateralRequest", "category": "post-trade", "description": "Request collateral action"},
    "AY": {"name": "CollateralAssignment", "category": "post-trade", "description": "Assign collateral"},
    "BB": {"name": "CollateralResponse", "category": "post-trade", "description": "Response to collateral request"},
    "BA": {"name": "CollateralReport", "category": "post-trade", "description": "Report collateral position"},
    "BG": {"name": "CollateralInquiry", "category": "post-trade", "description": "Inquire about collateral"},
    "AX": {"name": "RequestForPositions", "category": "post-trade", "description": "Request position information"},
    "AO": {"name": "RequestForPositionsAck", "category": "post-trade", "description": "Acknowledge request for positions"},

    # Market Data
    "V": {"name": "MarketDataRequest", "category": "market-data", "description": "Request market data"},
    "W": {"name": "MarketDataSnapshotFullRefresh", "category": "market-data", "description": "Full snapshot of market data"},
    "X": {"name": "MarketDataIncrementalRefresh", "category": "market-data", "description": "Incremental update of market data"},
    "Y": {"name": "MarketDataRequestReject", "category": "market-data", "description": "Reject a market data request"},
    "i": {"name": "MassQuote", "category": "market-data", "description": "Submit mass quotes"},
    "b": {"name": "MassQuoteAcknowledgement", "category": "market-data", "description": "Acknowledge mass quote"},
    "S": {"name": "Quote", "category": "market-data", "description": "Submit a single quote"},
    "R": {"name": "QuoteRequest", "category": "market-data", "description": "Request a quote"},
    "AI": {"name": "QuoteStatusRequest", "category": "market-data", "description": "Request quote status"},
    "AJ": {"name": "QuoteStatusReport", "category": "market-data", "description": "Report quote status"},
    "Z": {"name": "QuoteCancel", "category": "market-data", "description": "Cancel one or more quotes"},
    "K": {"name": "BidRequest", "category": "market-data", "description": "Request a bid response"},
    "L": {"name": "BidResponse", "category": "market-data", "description": "Respond to a bid request"},

    # Security & Reference Data
    "c": {"name": "SecurityDefinitionRequest", "category": "reference-data", "description": "Request security definition"},
    "d": {"name": "SecurityDefinition", "category": "reference-data", "description": "Security definition response"},
    "e": {"name": "SecurityStatusRequest", "category": "reference-data", "description": "Request status of a security"},
    "f": {"name": "SecurityStatus", "category": "reference-data", "description": "Security status response"},
    "o": {"name": "RegistrationInstructions", "category": "reference-data", "description": "Provide registration instructions"},
    "p": {"name": "RegistrationInstructionsResponse", "category": "reference-data", "description": "Response to registration instructions"},
    "v": {"name": "SecurityListRequest", "category": "reference-data", "description": "Request list of securities"},
    "y": {"name": "SecurityList", "category": "reference-data", "description": "List of securities"},
    "x": {"name": "SecurityTypeRequest", "category": "reference-data", "description": "Request security types"},
    "w": {"name": "SecurityTypes", "category": "reference-data", "description": "Response listing security types"},
    "BP": {"name": "DerivativeSecurityListRequest", "category": "reference-data", "description": "Request derivative security list"},
    "BQ": {"name": "DerivativeSecurityList", "category": "reference-data", "description": "Derivative security list response"},

    # Business Messages
    "j": {"name": "BusinessMessageReject", "category": "business", "description": "Reject an application-level message"},
    "n": {"name": "XMLMessage", "category": "business", "description": "Convey XML data within FIX"},
    "B": {"name": "News", "category": "business", "description": "Convey news/text"},
    "C": {"name": "Email", "category": "business", "description": "Convey an email"},

    # Network / Counterparty
    "BC": {"name": "NetworkCounterpartySystemStatusRequest", "category": "network", "description": "Request counterparty system status"},
    "BD": {"name": "NetworkCounterpartySystemStatusResponse", "category": "network", "description": "Counterparty system status response"},
    "BJ": {"name": "UserRequest", "category": "network", "description": "User management request"},
    "BK": {"name": "UserResponse", "category": "network", "description": "User management response"},
}


# ─── FIX 5.0 SP2 Fields ─────────────────────────────────────────────
# Maps tag number (int) → metadata dict.
#   name        : canonical field name
#   type        : FIX data type (STRING, PRICE, QTY, …)
#   description : functional description of the field

FIELDS: Dict[int, Dict[str, str]] = {
    1: {"name": "Account", "type": "STRING", "description": "Account mnemonic as agreed between buy and sell sides"},
    6: {"name": "AvgPx", "type": "PRICE", "description": "Calculated average price of all fills on this order"},
    7: {"name": "BeginSeqNo", "type": "SEQNUM", "description": "Message sequence number of first message in range to be resent"},
    8: {"name": "BeginString", "type": "STRING", "description": "Identifies beginning of new message and protocol version. ALWAYS first field. For FIXT: FIXT.1.1"},
    9: {"name": "BodyLength", "type": "LENGTH", "description": "Message length in bytes, forward to the CheckSum field"},
    10: {"name": "CheckSum", "type": "STRING", "description": "Three byte, simple checksum. ALWAYS last field"},
    11: {"name": "ClOrdID", "type": "STRING", "description": "Unique identifier for order as assigned by institution"},
    14: {"name": "CumQty", "type": "QTY", "description": "Total number of shares filled"},
    15: {"name": "Currency", "type": "CURRENCY", "description": "Identifies currency used for price"},
    16: {"name": "EndSeqNo", "type": "SEQNUM", "description": "Message sequence number of last message in range to be resent"},
    17: {"name": "ExecID", "type": "STRING", "description": "Unique identifier of execution message"},
    18: {"name": "ExecInst", "type": "MULTIPLECHARVALUE", "description": "Instructions for order handling on exchange/market"},
    20: {"name": "ExecTransType", "type": "CHAR", "description": "Identifies transaction type. Deprecated in FIX 5.0"},
    21: {"name": "HandlInst", "type": "CHAR", "description": "Instructions for order handling on broker's side"},
    22: {"name": "SecurityIDSource", "type": "STRING", "description": "Identifies class or source of the SecurityID value"},
    25: {"name": "IOIQltyInd", "type": "CHAR", "description": "Relative quality of indication"},
    30: {"name": "LastMkt", "type": "EXCHANGE", "description": "Market of execution for last fill"},
    31: {"name": "LastPx", "type": "PRICE", "description": "Price of last fill"},
    32: {"name": "LastQty", "type": "QTY", "description": "Quantity of last fill"},
    34: {"name": "MsgSeqNum", "type": "SEQNUM", "description": "Integer message sequence number"},
    35: {"name": "MsgType", "type": "STRING", "description": "Defines message type. Always third field in message"},
    36: {"name": "NewSeqNo", "type": "SEQNUM", "description": "New sequence number"},
    37: {"name": "OrderID", "type": "STRING", "description": "Unique identifier for order as assigned by sell-side"},
    38: {"name": "OrderQty", "type": "QTY", "description": "Quantity ordered"},
    39: {"name": "OrdStatus", "type": "CHAR", "description": "Identifies current status of order"},
    40: {"name": "OrdType", "type": "CHAR", "description": "Order type"},
    41: {"name": "OrigClOrdID", "type": "STRING", "description": "ClOrdID of the previous non-rejected order when canceling or replacing"},
    43: {"name": "PossDupFlag", "type": "BOOLEAN", "description": "Indicates possible retransmission of message with this sequence number"},
    44: {"name": "Price", "type": "PRICE", "description": "Price per unit of quantity"},
    45: {"name": "RefSeqNum", "type": "SEQNUM", "description": "Reference message sequence number"},
    47: {"name": "Rule80A", "type": "CHAR", "description": "Deprecated in FIX 5.0 – see OrderCapacity(528)"},
    48: {"name": "SecurityID", "type": "STRING", "description": "Security identifier value"},
    49: {"name": "SenderCompID", "type": "STRING", "description": "Assigned value used to identify firm sending message"},
    50: {"name": "SenderSubID", "type": "STRING", "description": "Assigned value used to identify specific message originator"},
    52: {"name": "SendingTime", "type": "UTCTIMESTAMP", "description": "Time of message transmission (always expressed in UTC)"},
    54: {"name": "Side", "type": "CHAR", "description": "Side of order"},
    55: {"name": "Symbol", "type": "STRING", "description": "Ticker symbol. Common, human understood representation of the security"},
    56: {"name": "TargetCompID", "type": "STRING", "description": "Assigned value used to identify receiving firm"},
    57: {"name": "TargetSubID", "type": "STRING", "description": "Assigned value used to identify specific individual or unit intended to receive message"},
    58: {"name": "Text", "type": "STRING", "description": "Free format text string"},
    59: {"name": "TimeInForce", "type": "CHAR", "description": "Specifies how long the order remains in effect"},
    60: {"name": "TransactTime", "type": "UTCTIMESTAMP", "description": "Timestamp when the business transaction represented by message occurred"},
    63: {"name": "SettlType", "type": "STRING", "description": "Indicates order settlement period"},
    64: {"name": "SettlDate", "type": "LOCALMKTDATE", "description": "Specific date of trade settlement"},
    65: {"name": "SymbolSfx", "type": "STRING", "description": "Additional information about the security"},
    75: {"name": "TradeDate", "type": "LOCALMKTDATE", "description": "Indicates date of trade referenced in this message"},
    76: {"name": "ExecBroker", "type": "STRING", "description": "Identifies executing / give-up broker"},
    77: {"name": "PositionEffect", "type": "CHAR", "description": "Indicates whether resulting position should be an opening or closing position"},
    97: {"name": "PossResend", "type": "BOOLEAN", "description": "Indicates that message may contain information that has been sent under another sequence number"},
    98: {"name": "EncryptMethod", "type": "INT", "description": "Method of encryption"},
    99: {"name": "StopPx", "type": "PRICE", "description": "Price per unit of quantity for stop orders"},
    100: {"name": "ExDestination", "type": "EXCHANGE", "description": "Execution destination as defined by institution"},
    102: {"name": "CxlRejReason", "type": "INT", "description": "Code to identify reason for cancel rejection"},
    103: {"name": "OrdRejReason", "type": "INT", "description": "Code to identify reason for order rejection"},
    108: {"name": "HeartBtInt", "type": "INT", "description": "Heartbeat interval (seconds)"},
    109: {"name": "ClientID", "type": "STRING", "description": "Firm identifier used in third party-Loss transactions"},
    110: {"name": "MinQty", "type": "QTY", "description": "Minimum quantity of an order to be executed"},
    111: {"name": "MaxFloor", "type": "QTY", "description": "Maximum number of shares within an order to be shown on the exchange floor"},
    112: {"name": "TestReqID", "type": "STRING", "description": "Identifier included in Test Request message to be returned in resulting Heartbeat"},
    122: {"name": "OrigSendingTime", "type": "UTCTIMESTAMP", "description": "Original time of message transmission when resending"},
    123: {"name": "GapFillFlag", "type": "BOOLEAN", "description": "Indicates that the Sequence Reset message is replacing administrative or application messages"},
    126: {"name": "ExpireTime", "type": "UTCTIMESTAMP", "description": "Time/Date of order expiration"},
    127: {"name": "DKReason", "type": "CHAR", "description": "Reason for execution rejection"},
    128: {"name": "DeliverToCompID", "type": "STRING", "description": "Assigned value used to identify the firm targeted to receive message"},
    129: {"name": "DeliverToSubID", "type": "STRING", "description": "Assigned value used to identify specific individual or unit at target firm"},
    141: {"name": "ResetSeqNumFlag", "type": "BOOLEAN", "description": "Indicates that both sides of the FIX session should reset sequence numbers"},
    150: {"name": "ExecType", "type": "CHAR", "description": "Describes the specific execution report"},
    151: {"name": "LeavesQty", "type": "QTY", "description": "Quantity open for further execution"},
    152: {"name": "CashOrderQty", "type": "QTY", "description": "Specifies the approximate order quantity desired in total monetary units"},
    167: {"name": "SecurityType", "type": "STRING", "description": "Indicates type of security"},
    168: {"name": "EffectiveTime", "type": "UTCTIMESTAMP", "description": "Time the details within the message should take effect"},
    200: {"name": "MaturityMonthYear", "type": "MONTHYEAR", "description": "Month and Year of the maturity"},
    201: {"name": "PutOrCall", "type": "INT", "description": "Indicates whether an Option is a Put or Call"},
    202: {"name": "StrikePrice", "type": "PRICE", "description": "Strike price"},
    204: {"name": "CustomerOrFirm", "type": "INT", "description": "Used for different account types"},
    206: {"name": "OptAttribute", "type": "CHAR", "description": "Can be used for different identification of option"},
    207: {"name": "SecurityExchange", "type": "EXCHANGE", "description": "Market used to help identify a security"},
    211: {"name": "PegOffsetValue", "type": "FLOAT", "description": "Amount (usually in decimals) used to compute the bid or offer"},
    262: {"name": "MDReqID", "type": "STRING", "description": "Unique identifier for Market Data Request"},
    263: {"name": "SubscriptionRequestType", "type": "CHAR", "description": "Subscription Request Type"},
    264: {"name": "MarketDepth", "type": "INT", "description": "Depth of market for Book Snapshot / Incremental updates"},
    265: {"name": "MDUpdateType", "type": "INT", "description": "Specifies the type of Market Data update"},
    266: {"name": "AggregatedBook", "type": "BOOLEAN", "description": "Specifies whether or not book entries should be aggregated"},
    267: {"name": "NoMDEntryTypes", "type": "NUMINGROUP", "description": "Number of MDEntryType entries"},
    268: {"name": "NoMDEntries", "type": "NUMINGROUP", "description": "Number of entries in Market Data message"},
    269: {"name": "MDEntryType", "type": "CHAR", "description": "Type of Market Data entry"},
    270: {"name": "MDEntryPx", "type": "PRICE", "description": "Price of the Market Data Entry"},
    271: {"name": "MDEntrySize", "type": "QTY", "description": "Quantity represented by the Market Data Entry"},
    272: {"name": "MDEntryDate", "type": "UTCDATEONLY", "description": "Date of Market Data Entry"},
    273: {"name": "MDEntryTime", "type": "UTCTIMEONLY", "description": "Time of Market Data Entry"},
    274: {"name": "TickDirection", "type": "CHAR", "description": "Direction of the tick"},
    275: {"name": "MDMkt", "type": "EXCHANGE", "description": "Market of the MDEntry"},
    276: {"name": "QuoteCondition", "type": "MULTIPLESTRINGVALUE", "description": "Space-delimited list of conditions describing a quote"},
    277: {"name": "TradeCondition", "type": "MULTIPLESTRINGVALUE", "description": "Space-delimited list of conditions describing a trade"},
    278: {"name": "MDEntryID", "type": "STRING", "description": "Unique identifier for Market Data Entry"},
    279: {"name": "MDUpdateAction", "type": "CHAR", "description": "Type of Market Data update action"},
    280: {"name": "MDEntryRefID", "type": "STRING", "description": "Reference to MDEntryID of previous update"},
    # Parties (FIX 5.0 SP2)
    448: {"name": "PartyID", "type": "STRING", "description": "Party identifier/code"},
    447: {"name": "PartyIDSource", "type": "CHAR", "description": "Identifies class or source of PartyID"},
    452: {"name": "PartyRole", "type": "INT", "description": "Identifies the type or role of a PartyID"},
    453: {"name": "NoPartyIDs", "type": "NUMINGROUP", "description": "Number of PartyID entries"},
    # Order Capacity & Restrictions
    528: {"name": "OrderCapacity", "type": "CHAR", "description": "Designates the capacity of the firm placing the order"},
    529: {"name": "OrderRestrictions", "type": "MULTIPLESTRINGVALUE", "description": "Restrictions associated with an order"},
    # FIX 5.0 SP2 additions
    544: {"name": "CashMargin", "type": "CHAR", "description": "Identifies whether an order is a margin order or a cash order"},
    553: {"name": "Username", "type": "STRING", "description": "Userid or username"},
    554: {"name": "Password", "type": "STRING", "description": "Password or passphrase"},
    555: {"name": "NoLegs", "type": "NUMINGROUP", "description": "Number of legs in multi-leg order"},
    600: {"name": "LegSymbol", "type": "STRING", "description": "Multi-leg instrument's individual leg's Symbol"},
    # Strategy parameters
    847: {"name": "TargetStrategy", "type": "INT", "description": "Target strategy of the order"},
    848: {"name": "TargetStrategyParameters", "type": "STRING", "description": "Field to allow further specification of the TargetStrategy"},
    # Trade Reporting
    856: {"name": "TradeReportType", "type": "INT", "description": "Type of Trade Report"},
    880: {"name": "TrdMatchID", "type": "STRING", "description": "Identifier assigned to a trade by a matching system"},
    # Application versioning (FIX 5.0)
    1128: {"name": "ApplVerID", "type": "STRING", "description": "Specifies the service pack release being applied to message at application level"},
    1129: {"name": "CstmApplVerID", "type": "STRING", "description": "Specifies a custom extension to message being applied"},
    1137: {"name": "DefaultApplVerID", "type": "STRING", "description": "Specifies the default application version ID for a session"},
}


# ─── Enumerations for key fields ────────────────────────────────────
# Maps "FieldName(Tag)" → {value_code: human_meaning}

ENUMERATIONS: Dict[str, Dict[str, str]] = {
    "Side(54)": {
        "1": "Buy", "2": "Sell", "3": "BuyMinus", "4": "SellPlus",
        "5": "SellShort", "6": "SellShortExempt", "7": "Undisclosed",
        "8": "Cross", "9": "CrossShort", "A": "CrossShortExempt",
        "B": "AsDefined", "C": "Opposite", "D": "Subscribe",
        "E": "Redeem", "F": "Lend", "G": "Borrow",
    },
    "OrdType(40)": {
        "1": "Market", "2": "Limit", "3": "Stop", "4": "StopLimit",
        "5": "MarketOnClose", "6": "WithOrWithout", "7": "LimitOrBetter",
        "8": "LimitWithOrWithout", "9": "OnBasis", "A": "OnClose",
        "B": "LimitOnClose", "C": "ForexMarket", "D": "PreviouslyQuoted",
        "E": "PreviouslyIndicated", "F": "ForexLimit", "G": "ForexSwap",
        "H": "ForexPreviouslyQuoted", "I": "Funari", "J": "MarketIfTouched",
        "K": "MarketWithLeftOverAsLimit", "L": "PreviousFundValuationPoint",
        "M": "NextFundValuationPoint", "P": "Pegged", "Q": "CounterOrderSelection",
    },
    "TimeInForce(59)": {
        "0": "Day", "1": "GoodTillCancel", "2": "AtTheOpening",
        "3": "ImmediateOrCancel", "4": "FillOrKill", "5": "GoodTillCrossing",
        "6": "GoodTillDate", "7": "AtTheClose", "8": "GoodThroughCrossing",
        "9": "AtCrossing",
    },
    "OrdStatus(39)": {
        "0": "New", "1": "PartiallyFilled", "2": "Filled", "3": "DoneForDay",
        "4": "Canceled", "5": "Replaced", "6": "PendingCancel", "7": "Stopped",
        "8": "Rejected", "9": "Suspended", "A": "PendingNew", "B": "Calculated",
        "C": "Expired", "D": "AcceptedForBidding", "E": "PendingReplace",
    },
    "ExecType(150)": {
        "0": "New", "3": "DoneForDay", "4": "Canceled", "5": "Replaced",
        "6": "PendingCancel", "7": "Stopped", "8": "Rejected", "9": "Suspended",
        "A": "PendingNew", "B": "Calculated", "C": "Expired", "D": "Restated",
        "E": "PendingReplace", "F": "Trade", "G": "TradeCorrect",
        "H": "TradeCancel", "I": "OrderStatus", "J": "TradeInAClearingHold",
        "K": "TradeHasBeenReleasedToClearing", "L": "TriggeredOrActivatedBySystem",
    },
    "HandlInst(21)": {
        "1": "AutomatedExecutionNoIntervention",
        "2": "AutomatedExecutionInterventionOK",
        "3": "ManualOrder",
    },
    "EncryptMethod(98)": {
        "0": "None", "1": "PKCS", "2": "DES", "3": "PKCS_DES",
        "4": "PGP_DES", "5": "PGP_DES_MD5", "6": "PEM_DES_MD5",
    },
    "SubscriptionRequestType(263)": {
        "0": "Snapshot", "1": "SnapshotAndUpdates", "2": "DisablePreviousSnapshot",
    },
    "MDUpdateType(265)": {
        "0": "FullRefresh", "1": "IncrementalRefresh",
    },
    "MDEntryType(269)": {
        "0": "Bid", "1": "Offer", "2": "Trade", "3": "IndexValue",
        "4": "OpeningPrice", "5": "ClosingPrice", "6": "SettlementPrice",
        "7": "TradingSessionHighPrice", "8": "TradingSessionLowPrice",
        "9": "TradingSessionVWAPPrice", "A": "Imbalance", "B": "TradeVolume",
        "C": "OpenInterest",
    },
    "SecurityIDSource(22)": {
        "1": "CUSIP", "2": "SEDOL", "3": "QUIK", "4": "ISINNumber",
        "5": "RICCode", "6": "ISOCurrencyCode", "7": "ISOCountryCode",
        "8": "ExchangeSymbol", "9": "ConsolidatedTapeAssociation",
        "A": "BloombergSymbol", "B": "Wertpapier", "C": "Dutch",
        "D": "Valoren", "E": "Sicovam", "F": "Belgian", "G": "Common",
        "H": "ClearingHouseNumber", "I": "ISDAFPMLProductSpecification",
        "J": "OptionPriceReportingAuthority", "K": "ISDAFPMLProductURL",
        "L": "LetterOfCredit",
    },
    "SecurityType(167)": {
        "CS": "CommonStock", "PS": "PreferredStock", "REPO": "Repurchase",
        "FORWARD": "ForwardRateAgreement", "BUYSELL": "BuySellback",
        "SECLOAN": "SecuritiesLoan", "SECPLEDGE": "SecuritiesPledge",
        "FUT": "Future", "OPT": "Option", "WAR": "Warrant",
        "MF": "MutualFund", "MLEG": "MultilegInstrument",
        "FXSPOT": "FXSpot", "FXFWD": "FXForward", "FXSWAP": "FXSwap",
    },
    "PositionEffect(77)": {
        "C": "Close", "F": "FIFO", "O": "Open", "R": "Rolled",
    },
    "OrderCapacity(528)": {
        "A": "Agency", "G": "Proprietary", "I": "Individual",
        "P": "Principal", "R": "RisklessPrincipal", "W": "AgentForOtherMember",
    },
}


# ─── Data Types ──────────────────────────────────────────────────────
# Maps FIX data-type name → description dict.

DATA_TYPES: Dict[str, Dict[str, str]] = {
    "INT": {"description": "Sequence of digits without commas or decimals, may include leading sign"},
    "LENGTH": {"description": "Non-negative integer representing message length"},
    "NUMINGROUP": {"description": "Number of repeating group entries"},
    "SEQNUM": {"description": "Message sequence number (positive integer)"},
    "TAGNUM": {"description": "FIX field tag number (positive integer)"},
    "FLOAT": {"description": "Sequence of digits with optional decimal point and sign"},
    "QTY": {"description": "Float quantity value"},
    "PRICE": {"description": "Float price value (typically 2 decimal places)"},
    "PRICEOFFSET": {"description": "Float value representing a price offset"},
    "AMT": {"description": "Float value typically representing a monetary amount"},
    "PERCENTAGE": {"description": "Float value representing a percentage"},
    "CHAR": {"description": "Single character value"},
    "BOOLEAN": {"description": "Y = Yes/True, N = No/False"},
    "STRING": {"description": "Alphanumeric free format strings"},
    "MULTIPLECHARVALUE": {"description": "Space-delimited multiple character fields"},
    "MULTIPLESTRINGVALUE": {"description": "Space-delimited multiple string fields"},
    "CURRENCY": {"description": "ISO 4217 currency code (3 chars)"},
    "EXCHANGE": {"description": "ISO 10383 Market Identifier Code (4 chars)"},
    "UTCTIMESTAMP": {"description": "UTC date/time: YYYYMMDD-HH:MM:SS.sss"},
    "UTCTIMEONLY": {"description": "UTC time only: HH:MM:SS.sss"},
    "UTCDATEONLY": {"description": "UTC date only: YYYYMMDD"},
    "LOCALMKTDATE": {"description": "Local date: YYYYMMDD"},
    "MONTHYEAR": {"description": "Month-year: YYYYMM or YYYYMMDD or YYYYMMwN"},
    "DATA": {"description": "Raw data with separate length field"},
    "XMLDATA": {"description": "XML-encoded data"},
}


# ─── Components ──────────────────────────────────────────────────────
# Maps component name → {description, fields (list of tag ints)}.

COMPONENTS: Dict[str, Dict[str, Any]] = {
    "StandardHeader": {
        "description": "Required header for all FIX messages (FIXT 1.1)",
        "fields": [8, 9, 35, 49, 56, 34, 52, 43, 97, 122, 1128, 1129],
    },
    "StandardTrailer": {
        "description": "Required trailer for all FIX messages",
        "fields": [10],
    },
    "Instrument": {
        "description": "Identifies a security",
        "fields": [55, 65, 48, 22, 167, 200, 201, 202, 206, 207, 15],
    },
    "OrderQtyData": {
        "description": "Specifies the quantity of an order",
        "fields": [38, 152],
    },
    "Parties": {
        "description": "Identifies parties in a trade",
        "fields": [453, 448, 447, 452],
    },
    "SpreadOrBenchmarkCurveData": {
        "description": "Spread or benchmark curve data",
        "fields": [218, 220, 221, 222, 662, 663, 699],
    },
    "CommissionData": {
        "description": "Commission data",
        "fields": [12, 13, 479, 497],
    },
}
