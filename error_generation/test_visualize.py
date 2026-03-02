import random
from error_generation import KoreanErrorGenerator

def visualize_errors():
    generator = KoreanErrorGenerator(seed=42)
    
    test_cases = [
        "안녕하세요, 저는 딥러닝 연구원입니다.",
        "오늘 날씨가 참 좋습니다. 그래서 굳이 우산을 챙길 필요가 없어요.",
        "과반수 이상이 동의했습니다. 역전 앞에서 만나요.",
        "할머니께서 진지를 잡수신다. 저를 보셨나요?",
        "10시 30분에 만나요. 제 나이는 30살입니다.",
        "apple은 맛있다. 이건 팩트야.",
        "정말 좋은 시간이었어. 아, 배가 고프다. 밥 먹으러 가자.",
        "가능하면 빨리 왔으면 해. 나중에 밥이나 먹자."
    ]
    
    print("=" * 60)
    print("1. 랜덤 에러 주입 (Random Mixed Errors)")
    print("=" * 60)
    for i, seq in enumerate(test_cases, 1):
        erroneous = generator.apply_random_errors(seq, n_errors=2)
        print(f"[문장 {i}]")
        print(f"원문: {seq}")
        print(f"오류: {erroneous}\n")
        
    print("=" * 60)
    print("2. 특정 에러 범주 지정 테스트 (Targeted Errors)")
    print("=" * 60)
    
    target_tests = {
        "common_misspellings": "금세 비가 그칠 줄 알았는데, 며칠 동안 내내 오네요.",
        "grammar_remove": "나는 학교에 가서 공부를 합니다.",
        "grammar_addition": "밥을 먹고 잠을 잤다.",
        "word_order_errors": "어제 친구와 함께 영화를 재미있게 봤다.",
        "semantic_errors": "이 시계는 성능이 아주 우수합니다. 가격도 저렴해요.",
        "tense_errors": "내일은 날씨가 맑겠습니다.",
        "chat_style_errors": "정말 감사합니다. 나중에 또 만나요.",
        "honorific_errors": "할아버지, 밥 먹어. 내가 줄게.",
        "number_errors": "사과 세 개랑 배 하나 주세요.",
        "typing_language_errors": "파이썬(python) 프로그래밍은 재미있다."
    }
    
    for err_type, seq in target_tests.items():
        print(f"[{err_type}]")
        print(f"원문: {seq}")
        # 계속 섞어서 오류가 걸릴 때까지 시도 (일부 문장은 해당 오류 패턴이 없을 수 있으므로)
        result = seq
        for _ in range(20):
            res = generator.apply_single_error(seq, error_type=err_type)
            if res is not None and res != seq:
                result = res
                break
        
        print(f"오류: {result}")
        if result == seq:
            print("(해당 문장에 적용 가능한 패턴이 없거나 변형되지 않음)")
        print()

if __name__ == "__main__":
    visualize_errors()
