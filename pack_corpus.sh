#!/bin/bash
# 코퍼스 파일들을 비밀번호로 암호화하고 100MB 크기로 분할 압축합니다.

echo -n "압축에 사용할 비밀번호를 입력하세요: "
read -s PASSWORD
echo ""

echo "파일을 압축, 암호화 및 분할 중입니다 (이 작업은 다소 시간이 걸릴 수 있습니다)..."

# 분할된 기존 파일들이 있다면 안전하게 삭제
rm -f corpus/corpus.tar.gz.enc.*

# tar로 묶고 -> openssl로 aes-256 암호화 -> split으로 100m씩 분할
tar czf - corpus/sample_1g.jsonl corpus/val_50k.jsonl corpus/sample_10g.jsonl | \
    openssl enc -aes-256-cbc -salt -pbkdf2 -e -pass pass:"$PASSWORD" | \
    split -b 100M - corpus/corpus.tar.gz.enc.

if [ $? -eq 0 ]; then
    echo "완료되었습니다! 압축 및 분할된 파일들이 corpus/ 디렉토리에 'corpus.tar.gz.enc.aa, ab, ...' 형태로 생성되었습니다."
    echo "이제 Docker image를 빌드하실 수 있습니다."
else
    echo "오류가 발생했습니다."
    exit 1
fi
