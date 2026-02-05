from __future__ import annotations

import re

_UNITS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(mg|g|μg|ug|ml|mL|片|粒|丸|袋|支|滴|%|毫克|克|微克|毫升)", re.IGNORECASE)
_PUNCT_RE = re.compile(r"[，。；、,.;:：!！?？()（）\[\]{}<>《》“”\"'`~·\s]+")

def normalize_text(s: str) -> str:
    """用于 contains/匹配的轻归一化：去空格与常见标点，不做分词。"""
    s = s.strip()
    s = _PUNCT_RE.sub("", s)
    return s

def strip_dosage_noise(s: str) -> str:
    """
    去掉剂量/单位噪音，让 alias 匹配更稳：
      络活喜5mg -> 络活喜
      Norvasc 5 mg -> Norvasc
    """
    s = s.strip()
    s = _UNITS_RE.sub("", s)
    # 还可能有类似 “5mg?” “5mg/日” 等残留
    s = re.sub(r"\d+(?:\.\d+)?", "", s)
    return s

def normalize_for_match(s: str) -> str:
    """匹配专用：先去剂量噪音，再做基础归一化。"""
    return normalize_text(strip_dosage_noise(s))
