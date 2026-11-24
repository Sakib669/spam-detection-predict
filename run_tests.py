import sys
import unittest

if __name__ == "__main__":
    from test_main import (
        test_root_endpoint,
        test_health_endpoint,
        test_predict_ham_message,
        test_predict_spam_message,
        test_predict_empty_message_rejected,
        test_predict_missing_field_rejected,
    )

    tests = [
        ("Root Endpoint", test_root_endpoint),
        ("Health Endpoint", test_health_endpoint),
        ("Predict Ham", test_predict_ham_message),
        ("Predict Spam", test_predict_spam_message),
        ("Empty Message Validation", test_predict_empty_message_rejected),
        ("Missing Field Validation", test_predict_missing_field_rejected),
    ]

    passed = 0
    for name, test_fn in tests:
        try:
            test_fn()
            print(f"✓ PASS: {name}")
            passed += 1
        except Exception as e:
            print(f"✗ FAIL: {name} - {e}")

    print(f"\nTest Results: {passed}/{len(tests)} passed.")
    if passed != len(tests):
        sys.exit(1)
