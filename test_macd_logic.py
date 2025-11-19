#!/usr/bin/env python3
"""Test script pour vérifier la logique de couleur de l'histogramme MACD"""

# Simule des valeurs d'histogramme pour comprendre la logique
test_cases = [
    # (date, hist_curr, hist_prev, expected_color, reason)
    ("Aug 1", -0.5, -0.7, "maroon", "négatif qui monte (rebondit) = vif"),
    ("Aug 2", -0.4, -0.5, "maroon", "négatif qui monte (rebondit) = vif"),
    ("Aug 5", -0.3, -0.4, "maroon", "négatif qui monte (rebondit) = vif"),
    ("Aug 6", -0.6, -0.3, "red", "négatif qui descend = sombre"),
    ("Aug 7", -0.8, -0.6, "red", "négatif qui descend = sombre"),

    ("Sept 3", -0.5, -0.7, "maroon", "négatif qui monte"),
    ("Sept 4", -0.4, -0.5, "maroon", "négatif qui monte"),
    ("Sept 5", -0.6, -0.4, "red", "négatif qui descend"),
]

def get_hist_color(hist_curr, hist_prev):
    """Logique actuelle dans macd_analyzer.py"""
    if hist_curr > 0:
        if hist_curr > hist_prev:
            return 'lime'
        else:
            return 'green'
    else:
        # When negative: rebounding (less negative) = maroon (vif)
        #                descending (more negative) = red (sombre)
        if hist_curr < hist_prev:
            return 'red'      # descending = sombre (no signal)
        else:
            return 'maroon'   # rebounding = vif (signal)

print("Test de la logique MACD histogram color:")
print("=" * 80)

all_correct = True
for date, hist_curr, hist_prev, expected, reason in test_cases:
    actual = get_hist_color(hist_curr, hist_prev)
    status = "✓" if actual == expected else "✗"
    if actual != expected:
        all_correct = False

    print(f"{status} {date:10s}: curr={hist_curr:5.1f}, prev={hist_prev:5.1f} "
          f"→ {actual:6s} (expected: {expected:6s}) - {reason}")

print("=" * 80)
if all_correct:
    print("✓ Tous les tests passent!")
else:
    print("✗ Certains tests échouent - la logique doit être corrigée")

print("\nLogique actuelle:")
print("- hist < 0 et hist < hist_prev → red (sombre) - pas de signal")
print("- hist < 0 et hist >= hist_prev → maroon (vif) - SELL signal")
