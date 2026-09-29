from __future__ import annotations

import re
from typing import Iterable, List, Optional, Tuple


SUPPORTED_SUFFIXES = {
    'SH', 'SZ', 'BJ',
    'H', 'HK',
    'US', 'N', 'O', 'A', 'K', 'BAT',
    'JP', 'T',
    'KS', 'KQ',
    'CFE', 'CFF', 'SHFE', 'DCE', 'CZCE', 'INE', 'GFEX',
}
# CICC appendix 1 has the futures exchange ids for CFFEX/SHFE swapped in text.
# The actual gateway mapping is CFFEX -> F, SHFE -> S.
FUTURE_SUFFIX_EXCH_ID = {
    'CFE': 'F',
    'CFF': 'F',
    'SHFE': 'S',
    'DCE': 'D',
    'CZCE': 'Z',
    'INE': 'N',
    'GFEX': 'G',
}

_A_SHARE_SH_PREFIXES = ('60', '68', '90')
_A_SHARE_SZ_PREFIXES = ('00', '001', '002', '003', '300', '301')
_A_SHARE_BJ_PREFIXES = ('43', '83', '87', '92')
_US_SUFFIXES = {'US', 'N', 'O', 'A', 'K', 'BAT'}
_HK_SUFFIXES = {'H', 'HK'}
_JP_SUFFIXES = {'JP', 'T'}
_US_BARE_PATTERN = re.compile(r'^[A-Z][A-Z0-9_-]*$')
_CN_INDEX_FUTURE_RE = re.compile(r'^(IC|IF|IH|IM|IO|MO|HO)\d{4}(?:-[CP]-\d+)?$', re.IGNORECASE)
REQUEST_USAGE_SUBSCRIBE = 'subscribe'
REQUEST_USAGE_SNAPSHOT = 'snapshot'
_REQUEST_USAGES = {REQUEST_USAGE_SUBSCRIBE, REQUEST_USAGE_SNAPSHOT}
_SPACE_SUFFIX_CODE_RE = re.compile(r'^([A-Z0-9._-]+)(?:\s+[A-Z0-9._-]+)*(?:\s+EQUITY)?$', re.IGNORECASE)


def normalize_quote_ticker(ticker: str, *, default_us_suffix: str = 'US') -> str:
    raw = str(ticker or '').strip().upper()
    if not raw:
        return ''

    if '.' in raw:
        code, suffix = raw.rsplit('.', 1)
        code = code.strip().upper()
        suffix = suffix.strip().upper()
        if suffix not in SUPPORTED_SUFFIXES or not code:
            return raw
        if suffix in _HK_SUFFIXES and code.isdigit():
            code = code.lstrip('0') or '0'
        return f'{code}.{suffix}'

    if raw.isdigit():
        return _normalize_bare_digits(raw)

    if _CN_INDEX_FUTURE_RE.match(raw):
        return f'{raw}.CFE'

    if _US_BARE_PATTERN.match(raw):
        return f'{raw}.{str(default_us_suffix or "US").strip().upper()}'

    return raw


def normalize_many_quote_tickers(tickers: Iterable[str], *, default_us_suffix: str = 'US') -> List[str]:
    result: List[str] = []
    seen = set()
    for ticker in tickers:
        normalized = normalize_quote_ticker(ticker, default_us_suffix=default_us_suffix)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def resolve_quote_request_field_candidates(
    ticker: str,
    *,
    request_usage: str = REQUEST_USAGE_SUBSCRIBE,
) -> List[Tuple[str, str, str]]:
    normalized = str(ticker or '').strip().upper()
    if not normalized or '.' not in normalized:
        return []
    request_usage = _normalize_request_usage(request_usage)
    code, suffix = normalized.rsplit('.', 1)
    code = code.strip().upper()
    suffix = suffix.strip().upper()
    if not code:
        return []
    if suffix == 'SH':
        category = 'O' if code.isdigit() and len(code) == 8 else 'S'
        return [(category, '0', code.zfill(8 if category == 'O' else 6))]
    if suffix == 'SZ':
        category = 'O' if code.isdigit() and len(code) == 8 else 'S'
        return [(category, '1', code.zfill(8 if category == 'O' else 6))]
    if suffix in _HK_SUFFIXES:
        return [('S', 'H', code.zfill(5))]
    if suffix == 'BJ':
        return [('S', '7', code.zfill(6))]
    if suffix in _US_SUFFIXES:
        return [('S', 'US', code)]
    if suffix in FUTURE_SUFFIX_EXCH_ID:
        category = 'O' if _is_future_option_code(code) else 'F'
        return [(category, FUTURE_SUFFIX_EXCH_ID[suffix], code)]
    if suffix in _JP_SUFFIXES:
        return [_resolve_stock_request_fields(code, 'JP', request_usage=request_usage)]
    if suffix == 'KS':
        return [_resolve_stock_request_fields(code.zfill(6), 'KS', request_usage=request_usage)]
    if suffix == 'KQ':
        return [_resolve_stock_request_fields(code.zfill(6), 'KQ', request_usage=request_usage)]
    return []


def resolve_quote_request_fields(
    ticker: str,
    *,
    request_usage: str = REQUEST_USAGE_SUBSCRIBE,
    asset_class: str = 'auto',
) -> Optional[Tuple[str, str, str]]:
    normalized_asset_class = str(asset_class or 'auto').strip().lower()
    if normalized_asset_class in {'index', 'indx'}:
        return resolve_index_request_fields(ticker, request_usage=request_usage)
    if normalized_asset_class not in {'auto', 'option', 'equity', 'stock'}:
        raise ValueError(f'Unsupported asset_class: {asset_class!r}')
    candidates = resolve_quote_request_field_candidates(ticker, request_usage=request_usage)
    return candidates[0] if candidates else None


def resolve_index_request_fields(
    ticker: str,
    *,
    request_usage: str = REQUEST_USAGE_SNAPSHOT,
) -> Optional[Tuple[str, str, str]]:
    """Resolve a mainland/HK index ticker to the gateway index category.

    Six-digit ``.SH``/``.SZ`` tickers are ambiguous in the generic mapping:
    they can represent either stocks or indices.  Index callers should use
    this explicit resolver so that ``000852.SH`` becomes ``INDEX/0/000852``.
    """
    normalized = normalize_quote_ticker(ticker)
    if not normalized or '.' not in normalized:
        return None
    code, suffix = normalized.rsplit('.', 1)
    code = code.strip().upper()
    suffix = suffix.strip().upper()
    if not code:
        return None
    if suffix == 'SH':
        return ('INDEX', '0', code.zfill(6))
    if suffix == 'SZ':
        return ('INDEX', '1', code.zfill(6))
    if suffix == 'BJ':
        return ('INDEX', '7', code.zfill(6))
    if suffix in _HK_SUFFIXES:
        return ('INDEX', 'H', code.zfill(5))
    return None


def normalize_quote_response_stk_code(exch_id: str, stk_code: str) -> str:
    normalized_exch = str(exch_id or '').strip().upper()
    raw_code = str(stk_code or '').strip().upper()
    if not raw_code:
        return ''

    match = _SPACE_SUFFIX_CODE_RE.match(raw_code)
    if match and normalized_exch in {'JP', 'KS', 'KQ'}:
        return match.group(1)
    return raw_code


def _normalize_bare_digits(digits: str) -> str:
    if len(digits) == 6:
        if digits.startswith(_A_SHARE_SH_PREFIXES):
            return f'{digits}.SH'
        if digits.startswith(_A_SHARE_SZ_PREFIXES):
            return f'{digits}.SZ'
        if digits.startswith(_A_SHARE_BJ_PREFIXES):
            return f'{digits}.BJ'
    if 1 <= len(digits) <= 5:
        return f'{digits.lstrip("0") or "0"}.HK'
    return digits


def _is_future_option_code(code: str) -> bool:
    upper_code = str(code or '').strip().upper()
    return '-C-' in upper_code or '-P-' in upper_code


def _normalize_request_usage(request_usage: str) -> str:
    normalized = str(request_usage or REQUEST_USAGE_SUBSCRIBE).strip().lower()
    if normalized not in _REQUEST_USAGES:
        raise ValueError(f'Unsupported request_usage: {request_usage!r}')
    return normalized


def _resolve_stock_request_fields(
    code: str,
    exch_id: str,
    *,
    request_usage: str,
) -> Tuple[str, str, str]:
    normalized_code = str(code or '').strip().upper()
    normalized_exch = str(exch_id or '').strip().upper()
    # The current gateway expects JP/KS/KQ snapshot requests in "CODE EXCH Equity" form.
    if request_usage == REQUEST_USAGE_SNAPSHOT and normalized_exch in {'JP', 'KS', 'KQ'}:
        return 'S', normalized_exch, f'{normalized_code} {normalized_exch} Equity'
    return 'S', normalized_exch, normalized_code


__all__ = [
    'FUTURE_SUFFIX_EXCH_ID',
    'REQUEST_USAGE_SNAPSHOT',
    'REQUEST_USAGE_SUBSCRIBE',
    'SUPPORTED_SUFFIXES',
    'normalize_many_quote_tickers',
    'normalize_quote_ticker',
    'normalize_quote_response_stk_code',
    'resolve_index_request_fields',
    'resolve_quote_request_field_candidates',
    'resolve_quote_request_fields',
]
