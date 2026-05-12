"""Hardcoded keyword lists used by the prompt analyzer. Kept separate to keep
prompt.py under the per-file line budget."""

CONSTRAINT_KEYWORDS = ["MUST", "NEVER", "ALWAYS", "DO NOT", "PROHIBITED", "REQUIRED"]

AMBIGUOUS_PHRASES = [
    "as needed",
    "use your judgment",
    "use your judgement",
    "feel free to",
    "try to",
    "be helpful",
    "be concise",
    "when relevant",
    "if appropriate",
    "as appropriate",
    "if necessary",
]

EXTERNAL_CONTENT_HINTS = [
    "email", "emails",
    "document", "documents",
    "web page", "webpage", "web pages",
    "user message", "user input", "user query",
    "url", "link",
    "pdf",
    "html",
]

OVERRIDE_PROTECTION_HINTS = [
    "ignore previous instructions",
    "do not follow instructions",
    "do not obey instructions",
    "never follow instructions",
    "prompt injection",
    "instruction override",
    "treat all user input as data",
    "treat content as data",
]

OUTPUT_FORMAT_HINTS = [
    "json", "yaml", "markdown",
    "respond in", "reply in", "output format",
    "respond with", "return a", "format your",
    "schema:",
]

EXAMPLE_HINTS = [
    "example:", "examples:", "for example",
    "e.g.", "e.g,", "here is an example",
    "<example", "input:", "output:",
]

SCOPE_HINTS = [
    "refuse", "decline", "do not answer",
    "only respond", "only answer", "only handle",
    "outside the scope", "outside your scope", "not your role",
    "out of scope", "off topic", "off-topic",
    "stay on topic", "if asked about", "ignore requests",
]

PERSONA_HINTS = [
    "you are a ", "you are an ", "you are the ",
    "your role is", "act as ", "acting as ",
    "you serve as", "you specialise as", "you specialize as",
]

OVERCONFIDENT_HINTS = [
    "always know", "never fail", "know everything",
    "have access to all", "expert in everything",
    "always correct", "all answers", "100% accurate",
]

UNCERTAINTY_HINTS = [
    "i don't know", "i do not know", "unsure", "uncertain",
    "if unknown", "if unsure", "say so", "admit",
    "cannot answer", "not enough information",
]
