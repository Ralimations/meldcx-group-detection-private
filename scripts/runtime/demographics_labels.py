#!/usr/bin/env python3
"""Helpers for translating model outputs into demographics labels."""

from __future__ import annotations


def split_label_tokens(raw_text: str) -> set[str]:
    return {token.strip().lower() for token in str(raw_text).split(",") if token.strip()}


def body_age_from_attrs(
    attrs: dict[str, float],
    minor_labels: set[str],
    adult_labels: set[str],
    senior_labels: set[str],
) -> tuple[str, float] | None:
    best_label: str | None = None
    best_score = 0.0
    for output_label, label_tokens in (
        ("Minor", minor_labels),
        ("Adult", adult_labels),
        ("Senior", senior_labels),
    ):
        if not label_tokens:
            continue
        score = 0.0
        for attr_name, attr_score in attrs.items():
            if str(attr_name).strip().lower() in label_tokens:
                score = max(score, float(attr_score))
        if score > best_score:
            best_label = output_label
            best_score = score
    if best_label is None:
        return None
    return best_label, best_score


def body_gender_from_attrs(attrs: dict[str, float], min_conf: float = 0.5) -> tuple[str, float] | None:
    male_score = 0.0
    longhair_score = 0.0
    for attr_name, attr_score in attrs.items():
        key = str(attr_name).strip().lower()
        score = float(attr_score)
        if key in {"gender_male", "is_male", "male", "gender-male"}:
            male_score = max(male_score, score)
        elif key in {"has_longhair", "hair_length_long", "hair-length-long"}:
            longhair_score = max(longhair_score, score)
    if male_score <= 0.0 and longhair_score <= 0.0:
        return None
    if male_score > 0.5:
        return "Male", male_score
    if longhair_score >= float(min_conf):
        return "Female", longhair_score
    if male_score > 0.0:
        return "Male", male_score
    return None
