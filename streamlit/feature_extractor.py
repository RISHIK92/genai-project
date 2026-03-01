"""
Feature extraction module for exam questions.
Extracts the 25 TEXT_FEATURES expected by the XGBoost models from raw question text and answer options.
"""

import re
import numpy as np


# --- Keyword dictionaries for domain detection ---
ADVANCED_TERMS = {
    'calculus', 'derivative', 'integral', 'differentiate', 'integrate',
    'limit', 'convergence', 'divergence', 'series', 'sequence',
    'logarithm', 'exponential', 'asymptote', 'matrix', 'matrices',
    'determinant', 'eigenvalue', 'vector', 'complex number', 'imaginary',
}

ALGEBRA_TERMS = {
    'equation', 'variable', 'solve', 'simplify', 'expression',
    'factorise', 'factorize', 'expand', 'substitute', 'algebra',
    'polynomial', 'quadratic', 'linear', 'coefficient', 'constant',
    'inequality', 'simultaneous', 'formula', 'rearrange', 'term',
    'binomial', 'trinomial',
}

GEOMETRY_TERMS = {
    'angle', 'triangle', 'circle', 'rectangle', 'square', 'polygon',
    'perimeter', 'area', 'volume', 'parallel', 'perpendicular',
    'symmetry', 'rotation', 'reflection', 'translation', 'congruent',
    'similar', 'hypotenuse', 'pythagoras', 'radius', 'diameter',
    'circumference', 'tangent', 'chord', 'arc', 'sector', 'segment',
    'cuboid', 'cylinder', 'cone', 'sphere', 'prism',
}

STATS_TERMS = {
    'mean', 'median', 'mode', 'range', 'probability', 'average',
    'frequency', 'histogram', 'pie chart', 'bar chart', 'scatter',
    'correlation', 'data', 'sample', 'survey', 'tally',
    'cumulative', 'quartile', 'percentile', 'standard deviation',
    'variance', 'distribution', 'expected', 'outcome', 'event',
    'random', 'likelihood', 'proportion',
}

# LaTeX command patterns
LATEX_PATTERNS = [
    r'\\frac', r'\\times', r'\\div', r'\\sqrt', r'\\sum', r'\\int',
    r'\\lim', r'\\infty', r'\\alpha', r'\\beta', r'\\theta', r'\\pi',
    r'\\geq', r'\\leq', r'\\neq', r'\\approx', r'\\pm',
    r'\\sin', r'\\cos', r'\\tan', r'\\log', r'\\ln',
    r'\\begin', r'\\end', r'\\text', r'\\mathbf', r'\\mathrm',
    r'\\left', r'\\right', r'\\cdot', r'\\circ',
    r'\\\(', r'\\\)', r'\\\[', r'\\\]',
]

MATH_OPERATORS = set('+-×÷=<>≤≥≠±^*/∙·')


def _count_sentences(text):
    """Count sentences using common delimiters."""
    cleaned = re.sub(r'\\[(\[\])]', '', text)  # remove LaTeX delimiters
    sentences = re.split(r'[.!?]+', cleaned)
    sentences = [s.strip() for s in sentences if s.strip()]
    return max(1, len(sentences))


def _count_latex_commands(text):
    """Count LaTeX commands in text."""
    count = 0
    for pattern in LATEX_PATTERNS:
        count += len(re.findall(pattern, text))
    return count


def _count_math_operators(text):
    """Count math operators in text."""
    count = sum(1 for ch in text if ch in MATH_OPERATORS)
    # Also count text-based operators
    count += len(re.findall(r'\\times', text))
    count += len(re.findall(r'\\div', text))
    count += len(re.findall(r'\\pm', text))
    count += len(re.findall(r'\\cdot', text))
    return count


def _count_numbers(text):
    """Count distinct number sequences."""
    return len(re.findall(r'\d+\.?\d*', text))


def _has_terms(text, term_set):
    """Check if text contains any terms from the set."""
    text_lower = text.lower()
    return 1 if any(term in text_lower for term in term_set) else 0


def _vocab_richness(words):
    """Calculate vocabulary richness (unique words / total words)."""
    if len(words) == 0:
        return 0.0
    unique = set(w.lower() for w in words)
    return len(unique) / len(words)


def _text_complexity_score(word_count, avg_word_length, latex_count, math_ops, number_count):
    """
    Compute a weighted composite text complexity score.
    Approximate the score used in the training data.
    """
    score = 0.0
    # Longer words = more complex
    score += avg_word_length * 0.3
    # More LaTeX = more complex
    score += latex_count * 0.2
    # More math operators = more complex
    score += math_ops * 0.15
    # More numbers = slightly more complex
    score += number_count * 0.1
    # Shorter questions with dense math are complex
    if word_count > 0:
        score += (latex_count + math_ops) / word_count * 2.0
    return round(score, 4)


def extract_features(question_text, answer_a, answer_b, answer_c, answer_d,
                     num_misconceptions=0, subject_difficulty_tier=1, construct_frequency=1):
    """
    Extract all 25 TEXT_FEATURES from raw question input.
    
    Args:
        question_text: The question text (may contain LaTeX)
        answer_a, answer_b, answer_c, answer_d: The 4 answer options
        num_misconceptions: Number of misconceptions (0-4)
        subject_difficulty_tier: Subject difficulty tier (1-5)
        construct_frequency: Construct frequency count
    
    Returns:
        dict: All 25 features as a dictionary
    """
    text = question_text.strip()
    
    # --- Basic text stats ---
    words = text.split()
    word_count = len(words)
    text_length = len(text)
    sentence_count = _count_sentences(text)
    avg_word_length = np.mean([len(w) for w in words]) if words else 0.0
    
    # --- LaTeX features ---
    latex_command_count = _count_latex_commands(text)
    has_latex = 1 if latex_command_count > 0 else 0
    latex_density = round(latex_command_count / word_count, 4) if word_count > 0 else 0.0
    
    # --- Math features ---
    math_operator_count = _count_math_operators(text)
    number_count = _count_numbers(text)
    
    # --- Vocabulary ---
    vocab_rich = round(_vocab_richness(words), 4)
    
    # --- Complexity ---
    complexity = _text_complexity_score(word_count, avg_word_length, 
                                        latex_command_count, math_operator_count, number_count)
    
    # --- Answer features ---
    ans_a_len = len(answer_a.strip())
    ans_b_len = len(answer_b.strip())
    ans_c_len = len(answer_c.strip())
    ans_d_len = len(answer_d.strip())
    answer_lengths = [ans_a_len, ans_b_len, ans_c_len, ans_d_len]
    avg_answer_length = np.mean(answer_lengths)
    answer_length_variance = np.var(answer_lengths)
    
    # --- Domain detection ---
    full_text = f"{text} {answer_a} {answer_b} {answer_c} {answer_d}"
    has_advanced = _has_terms(full_text, ADVANCED_TERMS)
    has_algebra = _has_terms(full_text, ALGEBRA_TERMS)
    has_geometry = _has_terms(full_text, GEOMETRY_TERMS)
    has_stats = _has_terms(full_text, STATS_TERMS)
    
    # --- Misconception features ---
    has_misconception = 1 if num_misconceptions > 0 else 0
    
    return {
        "text_length": text_length,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_word_length": round(avg_word_length, 4),
        "latex_command_count": latex_command_count,
        "has_latex": has_latex,
        "latex_density": latex_density,
        "math_operator_count": math_operator_count,
        "number_count": number_count,
        "vocab_richness": vocab_rich,
        "text_complexity_score": complexity,
        "answer_a_length": ans_a_len,
        "answer_b_length": ans_b_len,
        "answer_c_length": ans_c_len,
        "answer_d_length": ans_d_len,
        "avg_answer_length": round(avg_answer_length, 4),
        "answer_length_variance": round(answer_length_variance, 4),
        "has_advanced_terms": has_advanced,
        "has_algebra_terms": has_algebra,
        "has_geometry_terms": has_geometry,
        "has_stats_terms": has_stats,
        "num_misconceptions": num_misconceptions,
        "has_misconception": has_misconception,
        "subject_difficulty_tier": subject_difficulty_tier,
        "construct_frequency": construct_frequency,
    }


# Ordered feature list (must match training)
TEXT_FEATURES = [
    "text_length", "word_count", "sentence_count", "avg_word_length",
    "latex_command_count", "has_latex", "latex_density",
    "math_operator_count", "number_count", "vocab_richness",
    "text_complexity_score",
    "answer_a_length", "answer_b_length", "answer_c_length", "answer_d_length",
    "avg_answer_length", "answer_length_variance",
    "has_advanced_terms", "has_algebra_terms",
    "has_geometry_terms", "has_stats_terms",
    "num_misconceptions", "has_misconception",
    "subject_difficulty_tier", "construct_frequency",
]
