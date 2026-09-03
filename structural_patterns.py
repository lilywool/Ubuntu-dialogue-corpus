import re

_MENU_WORD = r"[A-Z][\w&/'-]*"
_MENU_SEGMENT = rf"{_MENU_WORD}(?:\s(?:{_MENU_WORD}|and|of|the|for)){{0,3}}"
MENU_PATH_RE = re.compile(
    rf'\b{_MENU_SEGMENT}(?:\s*>\s*{_MENU_SEGMENT}){{1,}}'
)

# Keyboard shortcuts: "Ctrl+Alt+T", "Alt+F2", "Ctrl + Shift + Esc". Requires the
# first token to be a real modifier key so ordinary math ("2+2+2") doesn't match.
_MODIFIER_KEYS = ['ctrl', 'control', 'alt', 'altgr', 'shift', 'super', 'win',
                   'windows', 'cmd', 'command', 'meta', 'fn']
_OTHER_KEYS = ['tab', 'esc', 'escape', 'enter', 'return', 'space', 'spacebar',
               'delete', 'del', 'backspace', 'capslock', 'home', 'end', 'pgup',
               'pgdn', 'pageup', 'pagedown', 'insert', 'ins', 'prtsc', 'up',
               'down', 'left', 'right'] + [f'f{i}' for i in range(1, 13)]
_KEY_ALTERNATION = '|'.join(sorted(_MODIFIER_KEYS + _OTHER_KEYS, key=len, reverse=True))
KEYBOARD_SHORTCUT_RE = re.compile(
    r'\b(?:' + '|'.join(_MODIFIER_KEYS) + r')'
    r'\s*\+\s*(?:' + _KEY_ALTERNATION + r'|[a-z0-9])'
    r'(?:\s*\+\s*(?:' + _KEY_ALTERNATION + r'|[a-z0-9]))*\b',
    re.IGNORECASE,
)

# Absolute Unix paths rooted at a known top-level directory, e.g.
# "/etc/modprobe.d/sound", "/var/log/syslog" -- generalizes the bare '/etc',
# '/usr', etc. entries already in the `filesystem` lexicon category to full paths.
UNIX_PATH_RE = re.compile(
    r'(?<![\w./])/(?:etc|usr|var|home|dev|proc|boot|mnt|media|bin|sbin|opt|root|'
    r'tmp|lib|lib64|srv|run)(?:/[\w.\-]+)*'
)

# Full URLs and bare domain-like fragments (http://..., www...., something.com).
# (?<!@) on the bare-domain branch keeps this from double-counting the domain
# half of an email address ("user@example.com" already matches EMAIL_ADDRESS_RE
# below in full; without the lookbehind, "example.com" would *also* fire here).
URL_OR_DOMAIN_RE = re.compile(
    r'\b(?:https?://|www\.)[^\s<>"\')]+'
    r'|(?<!@)\b[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.(?:com|net|org|io|co|gov|edu|info|biz)\b',
    re.IGNORECASE,
)

# Email addresses. Contact info, not a technical-content signal -- see docstring.
EMAIL_ADDRESS_RE = re.compile(r'\b[\w.+-]+@[\w-]+\.[a-z]{2,}(?:\.[a-z]{2,})?\b', re.IGNORECASE)

# Phone numbers: (###) ###-####, ###-###-####, ###.###.####, optional leading +1.
# Contact info, not a technical-content signal -- see docstring.
PHONE_NUMBER_RE = re.compile(
    r'(?<!\d)(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)'
)

# IPv4 addresses, e.g. "192.168.1.1", "10.0.0.255". Each octet is bounded to
# 0-255 (rather than a lazy \d{1,3}) so this doesn't also swallow version-like
# dotted-decimal strings with out-of-range octets. Still overlaps in spirit with
# things like Ubuntu release versions ("12.04") -- those won't false-positive
# here since they're only two segments, not four.
IPV4_ADDRESS_RE = re.compile(
    r'\b(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\b'
)

# IPv6 addresses, both full ("2001:0db8:85a3:0000:0000:8a2e:0370:7334") and
# "::"-compressed forms ("::1", "fe80::1ff:fe23:4567:890a", "::"). Boundaries
# use (?<![:\w.]) / (?![:\w.]) instead of \b: a compressed address can start
# or end with ':' itself (a non-word char), where \b would only fire if the
# preceding context happened to be a word character -- backwards from what we
# want (preceded by whitespace/punctuation is the common case). Every branch
# below requires an exact group count or an explicit "::", so ordinary
# single-colon text (timestamps "12:34:56", MAC addresses
# "00:1A:2B:3C:4D:5E") doesn't have enough colons/groups to match any branch.
IPV6_ADDRESS_RE = re.compile(
    r'(?<![:\w.])(?:'
    r'(?:[A-Fa-f0-9]{1,4}:){7}[A-Fa-f0-9]{1,4}'                          # full form, 8 groups
    r'|(?:[A-Fa-f0-9]{1,4}:){1,7}:'                                      # 1:: through 7::
    r'|(?:[A-Fa-f0-9]{1,4}:){1,6}:[A-Fa-f0-9]{1,4}'
    r'|(?:[A-Fa-f0-9]{1,4}:){1,5}(?::[A-Fa-f0-9]{1,4}){1,2}'
    r'|(?:[A-Fa-f0-9]{1,4}:){1,4}(?::[A-Fa-f0-9]{1,4}){1,3}'
    r'|(?:[A-Fa-f0-9]{1,4}:){1,3}(?::[A-Fa-f0-9]{1,4}){1,4}'
    r'|(?:[A-Fa-f0-9]{1,4}:){1,2}(?::[A-Fa-f0-9]{1,4}){1,5}'
    r'|[A-Fa-f0-9]{1,4}:(?:(?::[A-Fa-f0-9]{1,4}){1,6})'
    r'|:(?:(?::[A-Fa-f0-9]{1,4}){1,7}|:)'                                # ::1, ::
    r')(?![:\w.])'
)

# SSN-shaped: the canonical dashed form (123-45-6789) and the bare 9-digit
# form with no separators -- the latter is what actually turned up in this
# corpus's has_digit review bucket. Contact/identity info -- anonymized,
# same treatment as PHONE_NUMBER_RE, not just tracked like UNIX_PATH_RE.
SSN_RE = re.compile(r'(?<!\d)(?:\d{3}-\d{2}-\d{4}|\d{9})(?!\d)')

# Long runs of only 0/1 -- pasted binary dumps, bit patterns, "ascii as
# binary" jokes. Not anonymized -- not identifying info -- just tracked,
# same treatment as UNIX_PATH_RE.
BINARY_DUMP_RE = re.compile(r'\b[01]{8,}\b')

# Long hex strings -- MD5/SHA-style checksums (plausibly md5sum output from
# ISO-verification chatter) and memory addresses from crash logs. Requires
# >=12 hex chars AND at least one "genuinely hex" char (2-9 or a-f), so this
# doesn't fire on hex-charset English words ("cafe", "deadbeef") or
# double-count a pure 0/1 run as both binary_dump and hex_dump. Not
# anonymized -- see BINARY_DUMP_RE.
HEX_DUMP_RE = re.compile(r'\b(?=[0-9A-Fa-f]*[2-9A-Fa-f])[0-9A-Fa-f]{12,}\b', re.IGNORECASE)

# Number+unit "shapes" -- a data rate, a duration, a resolution, etc. Not PII,
# so these are tracked (their own structural_matches category) rather than
# anonymized, same treatment as unix_absolute_path/binary_dump/hex_dump.
_QUANTITY_UNITS = [
    # Original units
    'mbps', 'mb/s', 'kbps', 'mbit', 'mbs', "mb's",
    'kbytes', 'kb', 'gb', 'tb', 'gig', 'meg', 'k',
    'ghz', 'mhz', 'hz',
    'kwh', 'kw', 'v',
    'fps',
    'bit', 'b',
    'min', 'mins', 'hr', 'hrs', 'days', 'day', 'months', 'month',
    'years', 'year', 'seconds', 'second', 'ms', 'sec', 's',
    'ft', 'yo', 'pt', 'g', 'x',
    'mm', 'millimeter', 'millimeters', 'cm', 'centimeter', 'centimeters',
    'lb', 'lbs', 'pound', 'pounds', 'kg', 'kgs', 'kilogram', 'kilograms',
    
    # Computing & Hardware extensions
    'tbps', 'gbps', 'mb/sec', 'kb/sec', 'gib', 'mib', 'kib', 
    'core', 'cores', 'thread', 'threads',
    
    # Time extensions
    'week', 'weeks',
    
    # Metric & Imperial Length
    'm', 'meter', 'meters', 'km', 'kilometer', 'kilometers',
    'in', 'inch', 'inches', 'yd', 'yard', 'yards', 'mile', 'miles',
    
    # Mass & Weight extensions
    'gram', 'grams', 'mg', 'milligram', 'milligrams', 'oz', 'ounce', 'ounces',
    
    # Volume & Liquid
    'ml', 'milliliter', 'milliliters', 'l', 'liter', 'liters', 
    'gal', 'gallon', 'gallons', 'qt', 'quart', 'quarts',
    
    # Electronics, Battery & Power
    'mah', 'ah', 'mw', 'ma', 'kv',
    
    # Speed & Pressure
    'mph', 'kph', 'knots', 'psi',
]
_QUANTITY_UNIT_ALT = '|'.join(re.escape(u) for u in sorted(set(_QUANTITY_UNITS), key=len, reverse=True))
QUANTITY_UNIT_RE = re.compile(rf"\b\d+(?:\.\d+)?(?:{_QUANTITY_UNIT_ALT})\b", re.IGNORECASE)

ORDINAL_RE = re.compile(r'\b\d+(?:st|nd|rd|th)\b', re.IGNORECASE)                    # 18th, 2nd, 3rd, 21st
TIME_OF_DAY_RE = re.compile(r'\b\d+(?:am|pm)\b', re.IGNORECASE)                       # 7am, 7pm -- kept
                                                                                        # separate from the unit list since 'am'/'pm' are
                                                                                        # also ordinary standalone English words
APPROX_RANGE_RE = re.compile(r"\b\d+'s\b")                                            # 20's, 100's, 1000's
ASPECT_RATIO_RE = re.compile(r'\b\d+x\d+\b', re.IGNORECASE)                           # 640x420, 1920x1080
VIDEO_RESOLUTION_RE = re.compile(r'\b(?:360|480|720|1080)p?\b')                       # your specific whitelist
REFERENCE_YEAR_RE = re.compile(r'\b(?:1[89]\d{2}|20(?:0\d|1[0-5]))\b')                # 1800-2015

STRUCTURAL_PATTERNS = {
    "menu_navigation_path": MENU_PATH_RE,
    "keyboard_shortcut": KEYBOARD_SHORTCUT_RE,
    "unix_absolute_path": UNIX_PATH_RE,
    "url_or_domain": URL_OR_DOMAIN_RE,
    "email_address": EMAIL_ADDRESS_RE,
    "phone_number": PHONE_NUMBER_RE,
    "ipv4_address": IPV4_ADDRESS_RE,
    "ipv6_address": IPV6_ADDRESS_RE,
    "ssn": SSN_RE,
    "binary_dump": BINARY_DUMP_RE,
    "hex_dump": HEX_DUMP_RE,
    "quantity_unit": QUANTITY_UNIT_RE,
    "ordinal": ORDINAL_RE,
    "time_of_day": TIME_OF_DAY_RE,
    "approx_range": APPROX_RANGE_RE,
    "aspect_ratio": ASPECT_RATIO_RE,
    "video_resolution": VIDEO_RESOLUTION_RE,
    "reference_year": REFERENCE_YEAR_RE,
}


def find_structural_matches(text, patterns=STRUCTURAL_PATTERNS):
    """Run every STRUCTURAL_PATTERNS regex against `text` and return a dict of
    category -> list of matched strings (categories with no matches are omitted).
    Pass the raw, un-lowercased text -- these patterns rely on punctuation and
    (for keyboard shortcuts) are case-insensitive internally."""
    hits = {}
    for category, pattern in patterns.items():
        found = pattern.findall(text)
        if found:
            hits[category] = found
    return hits