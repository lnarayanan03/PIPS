from __future__ import annotations

import json
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

from anthropic import Anthropic


def _load_env_file(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _extract_json(raw: str) -> Any:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    object_start = raw.find("{")
    object_end = raw.rfind("}")
    array_start = raw.find("[")
    array_end = raw.rfind("]")

    if array_start != -1 and array_end != -1 and (
        object_start == -1 or array_start < object_start
    ):
        raw = raw[array_start:array_end + 1]
    elif object_start != -1 and object_end != -1 and object_end > object_start:
        start = object_start
        end = object_end
        raw = raw[start:end + 1]

    return json.loads(raw)


def _normalize_keywords(
    sections: Dict[str, Dict[str, Any]],
    raw_keywords: Dict[str, Any],
) -> Dict[str, list[str]]:
    normalized = {}
    for section_id in sections:
        values = raw_keywords.get(section_id, [])
        if not isinstance(values, list):
            values = []
        keywords = []
        seen = set()
        for value in values:
            keyword = str(value).strip().lower()
            if keyword and keyword not in seen:
                keywords.append(keyword)
                seen.add(keyword)
        if not keywords:
            keywords = [section_id.replace("_", " ")]
        normalized[section_id] = keywords[:24]
    return normalized


def _normalize_keyword_list(
    section_id: str,
    values: Any,
) -> list[str]:
    if not isinstance(values, list):
        values = []
    keywords = []
    seen = set()
    for value in values:
        keyword = str(value).strip().lower()
        if keyword and keyword not in seen:
            keywords.append(keyword)
            seen.add(keyword)
    if not keywords:
        keywords = [section_id.replace("_", " ")]
    return keywords[:24]


def _fallback_keyword_parse(raw: str) -> list[str]:
    cleaned = raw.replace("```json", "").replace("```", "")
    candidates = []
    for line in cleaned.splitlines():
        line = line.strip().strip(",")
        line = re.sub(r"^\s*[-*\d.]+\s*", "", line)
        line = line.strip().strip('"').strip("'").strip()
        if line and not line.startswith("[") and not line.startswith("]"):
            candidates.append(line)
    if not candidates:
        candidates = re.findall(r'"([^"]{2,80})"', cleaned)
    return candidates


def generate_section_keywords(
    section_id: str,
    content: str,
    model_name: str,
) -> list[str]:
    _load_env_file()
    client = Anthropic()
    system = (
        "You build keyword routing indexes for enterprise assistant prompts. "
        "Given one section ID and its content, generate routing keywords for that section. "
        "Use only the content and section meaning, not any pre-existing keyword lists. "
        "Return only valid JSON: an array of exactly 12 short keyword phrases."
    )
    user = (
        f"Generate PageIndex routing keywords for section_id={section_id}. "
        "Include synonyms, product terms, customer wording, error phrases, and policy terms. "
        "Do not include explanations.\n\n"
        f"CONTENT:\n{content}"
    )
    resp = client.messages.create(
        model=model_name,
        max_tokens=800,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    raw = resp.content[0].text.strip()
    try:
        values = _extract_json(raw)
    except json.JSONDecodeError:
        values = _fallback_keyword_parse(raw)
    return _normalize_keyword_list(section_id, values)


def generate_llm_keywords(
    sections: Dict[str, Dict[str, Any]],
    model_name: str,
    path: str = "data/generated_keywords.json",
    existing_keywords: Dict[str, list[str]] | None = None,
) -> Dict[str, list[str]]:
    keywords = dict(existing_keywords or {})
    index_path = Path(path)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    for section_id, section in sections.items():
        if section_id in keywords and keywords[section_id]:
            continue
        print(f"    Generating keywords for {section_id}...")
        keywords[section_id] = generate_section_keywords(
            section_id,
            section["content"],
            model_name,
        )
        index_path.write_text(json.dumps(keywords, indent=2))

    return _normalize_keywords(sections, keywords)


def load_or_generate_llm_keywords(
    sections: Dict[str, Dict[str, Any]],
    model_name: str,
    path: str = "data/generated_keywords.json",
    rebuild: bool = False,
) -> Dict[str, list[str]]:
    index_path = Path(path)
    existing_keywords = {}
    if index_path.exists() and not rebuild:
        existing_keywords = json.loads(index_path.read_text())
        if all(section_id in existing_keywords for section_id in sections):
            return _normalize_keywords(sections, existing_keywords)

    keywords = generate_llm_keywords(
        sections,
        model_name,
        path=path,
        existing_keywords={} if rebuild else existing_keywords,
    )
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(keywords, indent=2))
    return keywords


def apply_keywords(
    sections: Dict[str, Dict[str, Any]],
    keyword_map: Dict[str, list[str]],
) -> Dict[str, Dict[str, Any]]:
    indexed_sections = deepcopy(sections)
    for section_id, keywords in keyword_map.items():
        if section_id in indexed_sections:
            indexed_sections[section_id]["keywords"] = keywords
    return indexed_sections
