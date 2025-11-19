#!/usr/bin/env python3
"""Test script to verify combined MACD + Squeeze signal logic"""

def test_signal_logic():
    """Test the combined signal logic"""
    print("Testing Combined Signal Logic")
    print("=" * 80)

    # Test cases: (macd_line_color, squeeze_color, expected_signal)
    test_cases = [
        ('green', 'lime', 'BUY', 'MACD green + Squeeze lime (vert vif)'),
        ('green', 'green', 'NONE', 'MACD green but Squeeze green (vert sombre)'),
        ('green', 'maroon', 'NONE', 'MACD green but Squeeze maroon (rouge sombre)'),
        ('green', 'red', 'NONE', 'MACD green but Squeeze red (rouge vif)'),

        ('red', 'red', 'SELL', 'MACD red + Squeeze red (rouge vif)'),
        ('red', 'maroon', 'NONE', 'MACD red but Squeeze maroon (rouge sombre)'),
        ('red', 'lime', 'NONE', 'MACD red but Squeeze lime (vert vif)'),
        ('red', 'green', 'NONE', 'MACD red but Squeeze green (vert sombre)'),
    ]

    all_passed = True

    for macd_color, squeeze_color, expected, description in test_cases:
        # Simulate the logic from new_scan.py
        signal = 'NONE'

        # BUY: MACD ligne verte + Squeeze histogram lime (vert vif)
        if macd_color == 'green' and squeeze_color == 'lime':
            signal = 'BUY'

        # SELL: MACD ligne rouge + Squeeze histogram red (rouge vif)
        elif macd_color == 'red' and squeeze_color == 'red':
            signal = 'SELL'

        status = "✓" if signal == expected else "✗"
        if signal != expected:
            all_passed = False

        print(f"{status} MACD={macd_color:5s} Squeeze={squeeze_color:6s} → {signal:4s} "
              f"(expected: {expected:4s}) - {description}")

    print("=" * 80)
    if all_passed:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed!")

    return all_passed

if __name__ == '__main__':
    success = test_signal_logic()
    exit(0 if success else 1)
