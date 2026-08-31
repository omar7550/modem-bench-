"""Run-record writer: one honest document, one declared redaction policy per record shape."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
import json
import os
from pathlib import Path
import secrets
from typing import Any, Callable, Iterator, Mapping
from weakref import WeakKeyDictionary

from .sealed import capture_reference, gated_sealed_roots

# --- the nested redactors ----------------------------------------------------------------

# Sandbox fields derived from the sealed capture. bits_sha256 is an off-budget correctness
# oracle; receiver_sha256 can be the content address of a file inside the sealed store.
SEALED_SANDBOX_REDACTIONS: tuple[str, ...] = (
    "iq_sha256",
    "bits_sha256",
    "receiver_sha256",
    "stdout_tail",
    "stderr_tail",
)

# The receiver source digest under every key name a shipped record spells it with.
SEALED_RECEIVER_DIGEST_KEYS: tuple[str, ...] = (
    "receiver_sha256",
    "source_sha256",
)

# Private framing truth: where the packet starts, payload length, sync-word distance.
SEALED_EVALUATOR_REDACTIONS: tuple[str, ...] = (
    "alignment_offset",
    "n_payload_bits",
    "sync_hamming",
)

# Dyadic BER grid. A 1/1000 grid failed: 125-byte payloads are exactly 1000 bits, so
# continued fractions recovered the denominator. Reduced denominators of k/64 divide 64;
# shipped payloads start at 256 bits, so no published value reduces to a payload-bit count.
SEALED_BER_GRID_DENOMINATOR = 64


def quantize_ber_to_grid(value: float) -> float:
    """Snap a BER onto the published dyadic grid (multiples of 1/64 are exact doubles)."""
    return round(float(value) * SEALED_BER_GRID_DENOMINATOR) / SEALED_BER_GRID_DENOMINATOR


#: Marker keys stamped by sealed redactors; only when=SEALED operations can produce them.
SEALED_REDACTION_MARKER = "redacted_for_sealed"
SEALED_QUANTIZATION_MARKER = "aligned_ber_rounded_for_sealed"
SEALED_REDACTION_MARKERS = frozenset({SEALED_REDACTION_MARKER, SEALED_QUANTIZATION_MARKER})


def sealed_safe_sandbox(result: Mapping[str, Any]) -> dict[str, Any]:
    """Strip a sealed capture's identity, and everything recovered from it, from the repo."""
    redacted = {key: None for key in SEALED_SANDBOX_REDACTIONS if key in result}
    return {
        **{key: value for key, value in result.items() if key != "bits_path"},
        **redacted,
        SEALED_REDACTION_MARKER: sorted(redacted),
    }


def sealed_safe_feedback(feedback: Mapping[str, Any] | None) -> dict[str, Any]:
    """Snap aligned_ber to the grid: the raw BER's denominator is the payload length."""
    published = dict(feedback or {})
    ber = published.get("aligned_ber")
    if isinstance(ber, (int, float)) and not isinstance(ber, bool):
        published["aligned_ber"] = quantize_ber_to_grid(ber)
        published[SEALED_QUANTIZATION_MARKER] = True
    return published


def sealed_safe_evaluator_internal(internal: Mapping[str, Any] | None) -> dict[str, Any]:
    """Blank the private framing truth; packet_success survives."""
    blanked = dict(internal or {})
    redacted = sorted(name for name in SEALED_EVALUATOR_REDACTIONS if name in blanked)
    for name in redacted:
        blanked[name] = None
    blanked[SEALED_REDACTION_MARKER] = redacted
    return blanked


def sealed_safe_evaluator(evaluated: Mapping[str, Any]) -> dict[str, Any]:
    """Both halves of an evaluator result at once, for records that carry the whole thing."""
    return {
        **evaluated,
        "feedback": sealed_safe_feedback(evaluated.get("feedback")),
        "internal": sealed_safe_evaluator_internal(evaluated.get("internal")),
    }


# A policy stores the redactor name; the writer resolves it here.
REDACTORS: dict[str, Callable[[Mapping[str, Any]], dict[str, Any]]] = {
    "sandbox": sealed_safe_sandbox,
    "evaluator": sealed_safe_evaluator,
    "evaluator_feedback": sealed_safe_feedback,
    "evaluator_internal": sealed_safe_evaluator_internal,
}


# --- the policy vocabulary ---------------------------------------------------------------


class When(StrEnum):
    """When an operation applies."""

    ALWAYS = "always"
    SEALED = "sealed"


ALWAYS = When.ALWAYS
SEALED = When.SEALED


class RecordPolicyError(RuntimeError):
    """A policy does not fit the document it was applied to."""


def _require_mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RecordPolicyError(f"{where} is {type(value).__name__}, not a mapping")
    return value


def _require_keys(
    section: Mapping[str, Any], keys: tuple[str, ...], op: str, at: tuple[str, ...]
) -> None:
    missing = [key for key in keys if key not in section]
    if missing:
        location = ".".join(at) or "<root>"
        raise RecordPolicyError(
            f"{op} names {missing} at {location}, which the record does not carry: "
            "the field was renamed or removed and this redaction silently stopped applying"
        )


@dataclass(frozen=True, kw_only=True)
class Drop:
    """Remove keys entirely; tolerant of absent keys (dropped keys are conditional)."""

    keys: tuple[str, ...]
    at: tuple[str, ...] = ()
    when: When = ALWAYS

    def apply(self, section: Mapping[str, Any], **_: Any) -> dict[str, Any]:
        return {key: value for key, value in section.items() if key not in self.keys}


@dataclass(frozen=True, kw_only=True)
class Null:
    """Replace a named set of keys with None, keeping the keys visible."""

    keys: tuple[str, ...]
    at: tuple[str, ...] = ()
    when: When = SEALED

    def apply(self, section: Mapping[str, Any], **_: Any) -> dict[str, Any]:
        _require_keys(section, self.keys, "Null", self.at)
        return {**section, **{key: None for key in self.keys}}


@dataclass(frozen=True, kw_only=True)
class Substitute:
    """Replace one value with a template rendered against the write's context."""

    key: str
    template: str
    at: tuple[str, ...] = ()
    when: When = SEALED

    def apply(
        self, section: Mapping[str, Any], *, context: Mapping[str, Any], **_: Any
    ) -> dict[str, Any]:
        _require_keys(section, (self.key,), "Substitute", self.at)
        try:
            rendered = self.template.format_map(_StrictContext(context))
        except KeyError as exc:
            raise RecordPolicyError(
                f"Substitute for {self.key!r} needs context key {exc.args[0]!r}"
            ) from None
        return {**section, self.key: rendered}


class _StrictContext(dict):
    """format_map source refusing missing, empty, or non-string substitutions."""

    def __missing__(self, key: str) -> Any:
        raise KeyError(key)

    def __getitem__(self, key: str) -> Any:
        value = super().__getitem__(key)
        if not isinstance(value, str) or not value:
            raise RecordPolicyError(
                f"substitution context {key!r} is {value!r}, which would publish a "
                "placeholder where an opaque reference belongs"
            )
        return value


@dataclass(frozen=True, kw_only=True)
class Constant:
    """Flip one value to a fixed one."""

    key: str
    value: Any
    at: tuple[str, ...] = ()
    when: When = SEALED

    def apply(self, section: Mapping[str, Any], **_: Any) -> dict[str, Any]:
        _require_keys(section, (self.key,), "Constant", self.at)
        return {**section, self.key: self.value}


@dataclass(frozen=True, kw_only=True)
class Redact:
    """Apply a named nested redactor (see :data:`REDACTORS`) to one sub-document."""

    key: str
    redactor: str
    at: tuple[str, ...] = ()
    when: When = SEALED

    def apply(
        self,
        section: Mapping[str, Any],
        *,
        redactors: Mapping[str, Callable[[Mapping[str, Any]], dict[str, Any]]],
        **_: Any,
    ) -> dict[str, Any]:
        _require_keys(section, (self.key,), "Redact", self.at)
        try:
            redactor = redactors[self.redactor]
        except KeyError:
            raise RecordPolicyError(
                f"no redactor named {self.redactor!r}; known: {sorted(redactors)}"
            ) from None
        target = _require_mapping(
            section[self.key], f"{'.'.join((*self.at, self.key))} (redacted by {self.redactor})"
        )
        return {**section, self.key: redactor(target)}


Operation = Drop | Null | Substitute | Constant | Redact


@dataclass(frozen=True)
class RecordPolicy:
    """What one record shape publishes. Declared identity keys must be nulled (checked at construction)."""

    name: str
    operations: tuple[Operation, ...] = ()
    identity: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(set(self.identity)) != len(self.identity):
            raise RecordPolicyError(f"policy {self.name}: duplicate identity keys")
        nulled: set[str] = set()
        for operation in self.operations:
            if isinstance(operation, Null) and operation.when is SEALED and operation.at == ():
                nulled.update(operation.keys)
        unguarded = [key for key in self.identity if key not in nulled]
        if unguarded:
            raise RecordPolicyError(
                f"policy {self.name} declares identity keys it never nulls: {unguarded}"
            )

    def identity_fields(self, **values: Any) -> dict[str, Any]:
        """The identity half of a document: exactly the declared keys."""
        supplied = set(values)
        declared = set(self.identity)
        if supplied != declared:
            raise RecordPolicyError(
                f"policy {self.name} identity mismatch: "
                f"unexpected {sorted(supplied - declared)}, missing {sorted(declared - supplied)}"
            )
        return {key: values[key] for key in self.identity}

    def apply(
        self,
        document: Mapping[str, Any],
        *,
        sealed: bool,
        context: Mapping[str, Any] | None = None,
        redactors: Mapping[str, Callable[[Mapping[str, Any]], dict[str, Any]]] | None = None,
    ) -> dict[str, Any]:
        """Return the publishable form of ``document``. The input is never mutated."""
        resolved = REDACTORS if redactors is None else redactors
        supplied = dict(context or {})
        result = dict(document)
        for operation in self.operations:
            if operation.when is SEALED and not sealed:
                continue
            result = _edit(
                result,
                operation.at,
                # step is bound per iteration, not closed over
                lambda section, step=operation: step.apply(
                    section, context=supplied, redactors=resolved
                ),
            )
        return result


def _edit(
    document: Mapping[str, Any],
    at: tuple[str, ...],
    transform: Callable[[Mapping[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    """Apply ``transform`` to the mapping at ``at``, rebuilding the path above it."""
    section = _require_mapping(document, "the record")
    if not at:
        return transform(section)
    head, *rest = at
    if head not in section:
        raise RecordPolicyError(
            f"the record has no {head!r} section, so a policy operation addressed at "
            f"{'.'.join(at)} cannot apply"
        )
    return {**section, head: _edit(section[head], tuple(rest), transform)}


# --- sealedness derivation ------------------------------------------------------------------
#
# Sealedness is derived by asking modembench.sealed about a concrete subject; a
# caller-supplied bool is refused.


class SealedDecisionError(RecordPolicyError):
    """A document was about to be published somewhere its own values may not go."""


#: Answers live off-instance so object.__setattr__ has nothing to rewrite; only _derive
#: populates this, so object.__new__(Sealing) yields an object with no answers.
_SEALING_STATE: "WeakKeyDictionary[Sealing, dict[str, Any]]" = WeakKeyDictionary()

_SEALING_FIELDS = ("sealed", "subject", "describes", "capture_ref", "roots")

_NOT_DERIVED = (
    "a Sealing may only be derived from a run's subject "
    "(sealing_of_capture / sealing_of_location), never constructed: a sealedness "
    "that can be written down by hand is the caller-supplied boolean this type "
    "replaced"
)


class Sealing:
    """Whether the run a record describes is sealed. Derived, never constructed.

    No constructor, no subclassing, no __replace__, no settable attributes: each closed
    hole was a demonstrated forgery. Code that reaches into module state can still forge;
    the destination check (sealed_identity_in) does not depend on this type.
    """

    __slots__ = ("__weakref__",)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        raise SealedDecisionError(
            f"{cls.__name__} may not subclass Sealing: write_record accepts anything that "
            "passes isinstance(), so a subclass with a permissive constructor is a forged "
            "sealedness with the right type"
        )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise SealedDecisionError(_NOT_DERIVED)

    def __replace__(self, **changes: Any) -> Sealing:
        # copy.replace (3.13+) dispatches here, bypassing __init__
        raise SealedDecisionError(
            "a Sealing may not be replaced field by field: "
            f"replace(sealing, {', '.join(f'{k}=...' for k in changes) or '...'}) is how the "
            "previous version of this type was forged in one line"
        )

    def _answers(self) -> dict[str, Any]:
        try:
            return _SEALING_STATE[self]
        except KeyError:
            raise SealedDecisionError(_NOT_DERIVED) from None

    @property
    def sealed(self) -> bool:
        return self._answers()["sealed"]

    @property
    def subject(self) -> str:
        return self._answers()["subject"]

    @property
    def describes(self) -> str:
        return self._answers()["describes"]

    @property
    def capture_ref(self) -> str | None:
        return self._answers()["capture_ref"]

    @property
    def roots(self) -> tuple[Path, ...]:
        return self._answers()["roots"]

    def __repr__(self) -> str:
        try:
            answers = self._answers()
        except SealedDecisionError:
            return "<Sealing: not derived>"
        return (
            f"Sealing(sealed={answers['sealed']!r}, subject={answers['subject']!r}, "
            f"describes={answers['describes']!r}, capture_ref={answers['capture_ref']!r})"
        )

    def __bool__(self) -> bool:
        raise SealedDecisionError(
            "a Sealing is not a boolean; ask for its .sealed if you need the decision"
        )

    def describe(self) -> str:
        return f"{self.describes} {self.subject!r}"


def _derive(
    subject: str | os.PathLike[str], *, describes: str, token: Any, with_reference: bool
) -> Sealing:
    """The one derivation; the only writer of _SEALING_STATE."""
    roots = gated_sealed_roots(token)
    resolved = Path(os.path.realpath(os.fspath(subject)))
    sealed = any(_is_within(resolved, root) for root in roots)
    sealing = object.__new__(Sealing)
    _SEALING_STATE[sealing] = {
        "sealed": sealed,
        "subject": str(subject),
        "describes": describes,
        "capture_ref": capture_reference(subject) if sealed and with_reference else None,
        "roots": roots,
    }
    return sealing


def sealing_of_capture(capture_dir: str | os.PathLike[str], token: Any = None) -> Sealing:
    """Derive sealedness from the capture the run is about; carries the opaque capture reference."""
    return _derive(capture_dir, describes="the capture", token=token, with_reference=True)


def sealing_of_location(
    path: str | os.PathLike[str], *, describes: str, token: Any = None
) -> Sealing:
    """Derive sealedness from a location that is not one capture; carries no capture reference."""
    return _derive(path, describes=describes, token=token, with_reference=False)


# --- the backstop: what may not be published outside the sealed store -----------------------
#
# No value-pattern recognisers: a BER, a dollar cost and a duration are the same IEEE double,
# so a continued-fraction recogniser both false-positived on honest values and was defeated by
# one multiplication. Completeness is proved at policy declaration time instead
# (tests/test_isolation.py). What survives is grounded in the sealed roots, the
# directory names inside them, and the policies' own redaction key sets.

#: Capture dir name: sha256(canonical manifest)[:12]. Matched case-folded against directory
#: names that actually exist inside a sealed root, so folding cannot false-positive.
_CAPTURE_ID_LENGTH = 12
_HEX = frozenset("0123456789abcdefABCDEF")

#: Redacted keys recognised by name; dev records publish them on purpose, so the
#: recogniser is armed by a redaction marker rather than unconditional (see _scan).
RECOVERABLE_TRUTH_KEYS = (
    frozenset(SEALED_EVALUATOR_REDACTIONS)
    | frozenset(SEALED_SANDBOX_REDACTIONS)
    | frozenset(SEALED_RECEIVER_DIGEST_KEYS)
)

def _is_within(path: Path, parent: Path) -> bool:
    if path == parent:
        return True
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


#: One walked node: (where, text, is_key, stated). stated is the value a key leads to.
Node = tuple[str, str, bool, Any]


def _published_key_text(key: Any) -> str | None:
    """What json.dumps writes for this mapping key, or None if it refuses the key."""
    if isinstance(key, str):
        return key
    if isinstance(key, bool):
        return "true" if key else "false"
    if key is None:
        return "null"
    if isinstance(key, int):
        return int.__repr__(key)
    if isinstance(key, float):
        if key != key:
            return "NaN"
        if key == float("inf"):
            return "Infinity"
        if key == float("-inf"):
            return "-Infinity"
        return float.__repr__(key)
    return None


def _nodes(value: Any, at: str = "$") -> Iterator[Node]:
    """Yield mapping keys (as json spells them) and string leaves with their paths.

    Non-string leaves are not yielded: an int leaf can wear a capture id's shape (a byte
    count, a nanosecond delta), a false-positive channel. Non-string keys are yielded
    because their json text lands in the file.
    """
    if isinstance(value, Mapping):
        for key, item in value.items():
            here = f"{at}.{key}"
            text = _published_key_text(key)
            if text is not None:
                yield f"{here} (key)", text, True, item
            yield from _nodes(item, here)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _nodes(item, f"{at}[{index}]")
    elif isinstance(value, str):
        yield at, value, False, None


def _home_spellings(root: Path) -> tuple[str, ...]:
    """~ and $HOME abbreviations of a root under the home directory."""
    home = os.path.expanduser("~")
    text = str(root)
    if not home or home == os.sep or not text.startswith(home + os.sep):
        return ()
    tail = text[len(home) :]
    return (f"~{tail}", f"$HOME{tail}", f"${{HOME}}{tail}")


_CASE_INSENSITIVE: dict[str, bool] = {}


def _case_insensitive(root: Path) -> bool:
    """Probe whether a case-folded spelling of root opens the same directory.

    APFS is case-insensitive by default; an undecidable probe answers "insensitive" (fail closed).
    """
    text = str(root)
    cached = _CASE_INSENSITIVE.get(text)
    if cached is not None:
        return cached
    flipped = text.swapcase()
    try:
        answer = True if flipped == text or not os.path.exists(text) else os.path.exists(flipped)
    except OSError:  # pragma: no cover - a path the OS refuses to stat at all
        answer = True
    _CASE_INSENSITIVE[text] = answer
    return answer


def _path_candidates(value: str) -> tuple[Path, ...]:
    """Every filesystem location a string could be naming (~, $HOME, relative spellings)."""
    if not value or "\x00" in value:
        return ()
    expanded = os.path.expandvars(os.path.expanduser(value))
    found: list[Path] = []
    try:
        if os.path.isabs(expanded):
            found.append(Path(os.path.realpath(expanded)))
        elif os.sep in value or expanded != value or value.startswith(os.curdir):
            found.append(Path(os.path.realpath(os.path.abspath(expanded))))
    except (OSError, ValueError):  # pragma: no cover - unresolvable spellings
        return ()
    return tuple(dict.fromkeys(found))


def _root_containing(candidate: Path, roots: tuple[Path, ...]) -> Path | None:
    """The sealed root ``candidate`` lands strictly inside, honouring case folding."""
    for root in roots:
        if candidate != root and _is_within(candidate, root):
            return root
        if _case_insensitive(root):
            folded = Path(str(candidate).casefold())
            folded_root = Path(str(root).casefold())
            if folded != folded_root and _is_within(folded, folded_root):
                return root
    return None


def _markers(root: Path) -> tuple[tuple[str, bool], ...]:
    """(marker, casefold?) prefixes that spell root inside a longer string."""
    spellings = tuple(spelling + os.sep for spelling in (str(root), *_home_spellings(root)))
    found: list[tuple[str, bool]] = [(spelling, False) for spelling in spellings]
    if _case_insensitive(root):
        found.extend((spelling.casefold(), True) for spelling in spellings)
    return tuple(dict.fromkeys(found))


def _looks_like_a_capture_id(token: str) -> bool:
    return len(token) == _CAPTURE_ID_LENGTH and set(token) <= _HEX


def _hex_windows(value: str) -> frozenset[str]:
    """Every 12-char hex window, not just maximal runs; case-folded.

    Safe to over-generate: a candidate only matches if it names a capture directory that
    exists in the sealed store.
    """
    folded = value.casefold()
    return frozenset(
        window
        for start in range(len(folded) - _CAPTURE_ID_LENGTH + 1)
        if set(window := folded[start : start + _CAPTURE_ID_LENGTH]) <= _HEX
    )


def _sealed_capture_names(roots: tuple[Path, ...]) -> frozenset[str]:
    """Case-folded names of capture dirs one level down in each sealed root; stat-only, no token."""
    names: set[str] = set()
    for root in roots:
        try:
            splits = [entry for entry in root.iterdir() if entry.is_dir()]
        except OSError:
            continue
        for split in splits:
            try:
                names.update(entry.name.casefold() for entry in split.iterdir() if entry.is_dir())
            except OSError:
                continue
    return frozenset(names)


def _states_something(value: Any) -> bool:
    """Whether a redacted key carries content; empty containers are blanks like None/""."""
    if value is None:
        return False
    if isinstance(value, str):
        return value != ""
    if isinstance(value, (Mapping, list, tuple)):
        return len(value) > 0
    return True


def _scan(nodes: tuple[Node, ...], active: tuple[Path, ...]) -> dict[str, str]:
    """Apply the recognisers to a flattened document; the store is listed at most once.

    The restatement recogniser is armed by a redaction marker, not unconditional: dev
    records state the redacted keys on purpose.
    """
    found: dict[str, str] = {}
    id_candidates: list[tuple[str, str, bool]] = []
    restatements: list[tuple[str, str]] = []
    armed = False
    for where, value, is_key, stated in nodes:
        for candidate in _path_candidates(value):
            root = _root_containing(candidate, active)
            if root is not None:
                found[where] = (
                    f"names {value!r}, which resolves to {candidate}, a location inside "
                    f"the sealed store {root}"
                )
                break
        if where not in found:
            folded = value.casefold()
            for root in active:
                if any(marker in (folded if fold else value) for marker, fold in _markers(root)):
                    found[where] = f"spells the sealed store path {str(root)!r} inside its value"
                    break
        if where not in found:
            if _looks_like_a_capture_id(value):
                id_candidates.append((where, value, True))
            else:
                for token in _hex_windows(value):
                    id_candidates.append((where, token, False))
        if not is_key:
            continue
        if value in SEALED_REDACTION_MARKERS:
            armed = True
        elif value in RECOVERABLE_TRUTH_KEYS and _states_something(stated):
            restatements.append(
                (
                    where.removesuffix(" (key)"),
                    f"states {value!r} = {stated!r} in a document that a sealed policy has "
                    f"already stamped {sorted(SEALED_REDACTION_MARKERS)}: the redaction is "
                    "incomplete, and this key is private framing truth that no value-level "
                    "recogniser can see",
                )
            )
    if armed:
        for where, why in restatements:
            found.setdefault(where, why)
    if id_candidates:
        sealed_names = _sealed_capture_names(active)
        for where, value, whole in id_candidates:
            if value.casefold() in sealed_names:
                found[where] = (
                    f"is {value!r}, the id of a capture that lives in the sealed store"
                    if whole
                    else f"carries {value!r}, the id of a capture that lives in the sealed store"
                )
    return found


def sealed_identity_in(
    document: Mapping[str, Any], roots: tuple[Path, ...] | None = None
) -> dict[str, str]:
    """{where: why} for every value that names the sealed store; empty is publishable.

    Recognisers: paths resolving strictly inside a sealed root (with ~, $HOME and
    case-folded spellings), the root spelled inside a longer string, 12-hex values naming an
    existing sealed capture directory, and redacted keys restated in a marker-stamped
    document. Values that merely recover sealed truth arithmetically (a raw BER) cannot be
    recognised here; that property is proved at policy declaration time.
    """
    active = gated_sealed_roots() if roots is None else roots
    if not active:
        return {}
    return _scan(tuple(_nodes(document, "$")), active)


def sealed_identity_in_path(
    path: str | os.PathLike[str], roots: tuple[Path, ...] | None = None
) -> dict[str, str]:
    """The same recognisers applied to a destination path (run dirs are minted from ids)."""
    active = gated_sealed_roots() if roots is None else roots
    if not active:
        return {}
    text = os.fspath(path)
    nodes: tuple[Node, ...] = ((f"the destination path {text!r}", text, False, None),)
    return _scan(nodes, active)


def refuse_sealed_identity(
    document: Mapping[str, Any],
    *,
    path: Path,
    because: str,
    roots: tuple[Path, ...] | None = None,
) -> None:
    """Raise unless document may be published at path; checks both content and destination."""
    offences = sealed_identity_in(document, roots)
    offences.update(sealed_identity_in_path(path, roots))
    if offences:
        raise SealedDecisionError(
            f"refusing to publish {path}: {because}, and the document carries sealed "
            f"identity at {sorted(offences)} — {'; '.join(f'{k} {v}' for k, v in sorted(offences.items()))}. "
            "A document whose own values name the sealed store is not publishable outside it, "
            "whatever the writer believes about its sealedness."
        )


def _destination_is_inside_the_sealed_store(path: Path, roots: tuple[Path, ...]) -> bool:
    # ask about the parent: the file itself does not exist yet
    resolved = Path(os.path.realpath(os.fspath(path.parent)))
    return any(_is_within(resolved, root) for root in roots)


# --- the writer ---------------------------------------------------------------------------


def write_json_once(
    path: Path, value: Mapping[str, Any], *, description: str = "run artifact"
) -> Path:
    """Write one immutable JSON artifact atomically; differing existing content is a hard error.

    The sealed-identity backstop lives here, keyed on the destination, so every writer
    inherits it.
    """
    roots = gated_sealed_roots()
    if not _destination_is_inside_the_sealed_store(path, roots):
        refuse_sealed_identity(
            value,
            path=path,
            because="its destination is outside every sealed root",
            roots=roots,
        )
    content = canonical_line(value)
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"refusing to overwrite differing {description}: {path}")
        return path
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    with temporary.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return path


def write_record(
    path: Path,
    document: Mapping[str, Any],
    policy: RecordPolicy,
    *,
    sealing: Sealing,
    context: Mapping[str, Any] | None = None,
    redactors: Mapping[str, Callable[[Mapping[str, Any]], dict[str, Any]]] | None = None,
    description: str = "run artifact",
) -> dict[str, Any]:
    """Redact document under policy and write it once; returns the published form."""
    if not isinstance(sealing, Sealing):
        raise SealedDecisionError(
            f"write_record needs a derived Sealing, not {type(sealing).__name__}: derive it "
            "with records.sealing_of_capture(capture_dir, token) or "
            "records.sealing_of_location(path, describes=...). A caller-supplied boolean is "
            "the defect this parameter replaced — sealed identity reached the repository five "
            "times, and the fifth was a call site that simply passed the wrong one."
        )
    supplied = dict(context or {})
    if "capture_ref" in supplied:
        raise SealedDecisionError(
            "capture_ref is supplied by the Sealing, not by the caller: the opaque reference "
            "must come from the same derivation as the decision to substitute it"
        )
    if sealing.capture_ref is not None:
        supplied["capture_ref"] = sealing.capture_ref
    published = policy.apply(
        document, sealed=sealing.sealed, context=supplied, redactors=redactors
    )
    if not sealing.sealed:
        # catches a Sealing derived from the wrong subject
        refuse_sealed_identity(
            published,
            path=path,
            because=f"it was published as unsealed, derived from {sealing.describe()}",
            roots=sealing.roots,
        )
    write_json_once(path, published, description=description)
    return published


# --- the append-only trace ------------------------------------------------------------------
#
# Each event carries prev_sha256 of the previous line's exact bytes; the run record carries
# the head. The head lives beside the trace, so the chain is evidence against accidental loss
# and partial tampering only, not against an adversary with write access to the run dir.
class TraceChainError(RuntimeError):
    """A trace's hash chain does not hold: it was truncated, spliced or rewritten."""


TRACE_POLICY_VERSION = "modembench-trace-v1"

#: Fields the writer owns; a caller supplying one could forge the chain.
_TRACE_RESERVED = ("seq", "prev_sha256")


def canonical_line(value: Mapping[str, Any]) -> bytes:
    """The exact bytes one record occupies: sorted keys, no spaces, one trailing newline."""
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _trace_lines(path: Path) -> list[bytes]:
    """Byte-exact lines. Blank lines and a missing final newline are refused so the chain
    authenticates the file, not just the events."""
    if not path.exists():
        return []
    raw = path.read_bytes()
    if not raw:
        return []
    if not raw.endswith(b"\n"):
        raise TraceChainError(
            f"{path}: the trace does not end with a newline. Every event is written with one, "
            "so a missing terminal newline is a truncated final line."
        )
    lines = raw.split(b"\n")[:-1]
    for index, line in enumerate(lines):
        if not line:
            raise TraceChainError(
                f"{path}: line {index} is blank. The writer emits one event per line and never "
                "a blank one, so this is content the chain cannot account for."
            )
    return [line + b"\n" for line in lines]


def trace_head_sha256(path: Path) -> str | None:
    """The digest of the trace's last line, or ``None`` for an absent or empty trace."""
    lines = _trace_lines(path)
    return sha256(lines[-1]).hexdigest() if lines else None


def append_record(
    path: Path,
    event: Mapping[str, Any],
    policy: RecordPolicy,
    *,
    sealing: Sealing,
    context: Mapping[str, Any] | None = None,
    redactors: Mapping[str, Callable[[Mapping[str, Any]], dict[str, Any]]] | None = None,
    description: str = "trace event",
) -> dict[str, Any]:
    """Redact one event, chain it, append it; the backstop runs per event, not per file."""
    if not isinstance(sealing, Sealing):
        raise SealedDecisionError(
            f"append_record needs a derived Sealing, not {type(sealing).__name__}: derive it "
            "with records.sealing_of_capture(capture_dir, token). The trace is the fifth "
            "external artifact in the class that has leaked sealed identity five times, and "
            "the fifth leak was a call site that passed the wrong boolean."
        )
    reserved = [key for key in _TRACE_RESERVED if key in event]
    if reserved:
        raise RecordPolicyError(
            f"the trace writer owns {reserved}: an event that supplies its own prev_sha256 "
            "can forge the chain, which is the one thing the chain exists to prevent"
        )
    supplied = dict(context or {})
    if "capture_ref" in supplied:
        raise SealedDecisionError(
            "capture_ref is supplied by the Sealing, not by the caller: the opaque reference "
            "must come from the same derivation as the decision to substitute it"
        )
    if sealing.capture_ref is not None:
        supplied["capture_ref"] = sealing.capture_ref

    lines = _trace_lines(path)
    published = policy.apply(event, sealed=sealing.sealed, context=supplied, redactors=redactors)
    published = {
        **published,
        "seq": len(lines),
        "prev_sha256": sha256(lines[-1]).hexdigest() if lines else None,
    }

    roots = gated_sealed_roots()
    if not _destination_is_inside_the_sealed_store(path, roots):
        refuse_sealed_identity(
            published,
            path=path,
            because="its destination is outside every sealed root",
            roots=roots,
        )
    if not sealing.sealed:
        refuse_sealed_identity(
            published,
            path=path,
            because=f"it was published as unsealed, derived from {sealing.describe()}",
            roots=sealing.roots,
        )

    content = canonical_line(published)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    # O_APPEND writes land whole; fsync so the head is never durable without its line
    with path.open("ab") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    del description  # kept for symmetry with the other writers
    return published


def verify_trace_chain(path: Path, *, expected_head: str | None = None) -> dict[str, Any]:
    """Raise TraceChainError at the first line that does not chain."""
    lines = _trace_lines(path)
    previous: str | None = None
    for index, line in enumerate(lines):
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TraceChainError(f"{path}: line {index} is not JSON: {exc}") from None
        if not isinstance(event, dict):
            raise TraceChainError(f"{path}: line {index} is not an object")
        if event.get("seq") != index:
            raise TraceChainError(
                f"{path}: line {index} claims seq {event.get('seq')!r} — lines were removed, "
                "reordered or inserted"
            )
        if event.get("prev_sha256") != previous:
            raise TraceChainError(
                f"{path}: line {index} chains to {event.get('prev_sha256')!r} but the previous "
                f"line hashes to {previous!r} — the trace was rewritten at or before this line"
            )
        previous = sha256(line).hexdigest()
    if expected_head is not None and previous != expected_head:
        raise TraceChainError(
            f"{path}: head is {previous!r}, the record says {expected_head!r} — the trace was "
            "truncated or extended after the record was written"
        )
    return {
        "path": str(path),
        "events": len(lines),
        "head_sha256": previous,
        "trace_policy_version": TRACE_POLICY_VERSION,
    }
