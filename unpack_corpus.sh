#!/bin/bash
# 암호화 및 분할된 코퍼스 파일을 해제합니다. 실행하기 전에 비밀번호를 준비하세요.

cd /workspace/base-model-2 || exit 1

echo -n "압축 해제에 사용할 비밀번호를 입력하세요: "
read -s PASSWORD
echo ""

echo "파일을 복호화하고 압축을 해제하는 중입니다..."

# 분할된 파일들을 합쳐서 -> openssl로 aes-256 복호화 -> tar로 추출 (현재 폴더를 덮어씁니다.)
cat corpus/corpus.tar.gz.enc.* | \
    openssl enc -aes-256-cbc -salt -pbkdf2 -d -pass pass:"$PASSWORD" | \
    tar xzf - 

if [ $? -eq 0 ]; then
    echo "압축 해제가 완료되었습니다!"
    echo "선택: 불필요한 분할 압축 원본 파일(corpus.tar.gz.enc.*)을 삭제하시겠습니까? (y/N)"
    read -r DELETE_OPT
    if [[ "$DELETE_OPT" == "y" || "$DELETE_OPT" == "Y" ]]; then
        rm -f corpus/corpus.tar.gz.enc.*
        echo "분할 압축 원본 파일이 모두 삭제되었습니다."
    fi
else
    echo "오류가 발생했습니다. 비밀번호가 틀렸거나 파일이 손상되었을 수 있습니다."
    exit 1
fi
