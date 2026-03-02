import random
from collections import Counter
from error_generation import KoreanErrorGenerator

def run_distribution_test():
    gen = KoreanErrorGenerator(seed=100)
    
    # Text with diverse linguistic features
    text = (
        "안녕하세요, 저는 딥러닝 연구원입니다. 오늘 날씨가 참 좋습니다. "
        "그래서 굳이 우산을 챙길 필요가 없어요. 하지만 내일은 비가 올지 모르니 조심하세요. "
        "과반수 이상이 동의했습니다. 역전 앞에서 만나요. 가능하면 빨리 왔으면 해. "
        "정말 좋은 시간이었어. 아, 배가 고프다. 밥 먹으러 가자. "
        "할머니께서 진지를 잡수신다. 저를 보셨나요? "
        "10시 30분에 만나요. 제 나이는 30살입니다. "
        "apple은 맛있다. 이건 팩트야. "
    )
    
    counts = Counter()
    total_samples = 5000
    
    for _ in range(total_samples):
        # Apply a single error and trace which one was applied
        for _attempt in range(10):
            # Hack to get the selected error type
            [chosen_fn] = gen._rng.choices(gen._fns, weights=gen._weights, k=1)
            result = chosen_fn(text, gen._rng)
            if result is not None:
                # Find the name matching the function
                idx = gen._fns.index(chosen_fn)
                name = gen._names[idx]
                counts[name] += 1
                break
                
    print("=== Error Distribution over 5000 samples ===")
    for name, count in counts.most_common():
        print(f"{name:30}: {count} ({count/total_samples*100:.2f}%)")

if __name__ == "__main__":
    run_distribution_test()
