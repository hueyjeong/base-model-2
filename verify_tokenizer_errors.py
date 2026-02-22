import sys
import os
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from error_generation import KoreanErrorGenerator
from keyboard_tokenizer.keyboard_wrapper import KeyboardTokenizer

def main():
    print("Testing Keyboard Tokenizer with Error Generation...")
    
    # Initialize tokenizer
    tokenizer_path = os.path.join(os.path.dirname(__file__), "keyboard_tokenizer", "keyboard_tokenizer.json")
    tok = KeyboardTokenizer(tokenizer_path)
    
    # Initialize error generator
    gen = KoreanErrorGenerator(seed=42)
    
    sentences = [
        "굳이 그럴 필요가 없다.",
        "안녕하세요. 감사합니다.",
        "이것뿐이다.",
        "사흘 뒤에 만나자.",
        "곧 도착합니다.",
        "제가 했습니다."
    ]
    
    print("=" * 60)
    for sent in sentences:
        print(f"Original: {sent}")
        
        # Apply 2 errors
        errored = gen.apply_random_errors(sent, n_errors=2)
        print(f"Errored : {errored}")
        
        # Tokenize
        encoded = tok.encode(errored, add_special=False)
        
        # Decode
        decoded = tok.decode(encoded, skip_special=True)
        print(f"Decoded : {decoded}")
        
        # Compare
        if errored == decoded:
            print("Status  : ✓ MATCH")
        else:
            print("Status  : ✗ MISMATCH")
            print(f"Tokens  : {encoded}")
        print("-" * 60)

    # Hardcoded test for "안ㄴㅕㅇ하세요"
    hardcoded = "안ㄴㅕㅇ하세요"
    print(f"Original: 안녕하세요 (Hardcoded Test)")
    print(f"Errored : {hardcoded}")
    encoded = tok.encode(hardcoded, add_special=False)
    decoded = tok.decode(encoded, skip_special=True)
    print(f"Decoded : {decoded}")
    if hardcoded == decoded:
        print("Status  : ✓ MATCH")
    else:
        print("Status  : ✗ MISMATCH")
        print(f"Tokens  : {encoded}")
    print("=" * 60)

if __name__ == "__main__":
    main()
