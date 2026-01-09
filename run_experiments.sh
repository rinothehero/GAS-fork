#!/bin/bash

# GAS 순차 실험 스크립트
# 사용법: bash run_experiments.sh

# ANSI 색상 코드
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  GAS Sequential Experiments Runner${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 백업 생성
echo -e "${GREEN}[1/4] Creating backup of GAS_main.py...${NC}"
cp GAS_main.py GAS_main.py.backup
echo "Backup created: GAS_main.py.backup"
echo ""

# 실험 설정 정의 (예시)
declare -a experiments=(
    "exp1:use_resnet=False,split_alexnet='default',epochs=300,shard=8,alpha=0.9"
    "exp2:use_resnet=False,split_alexnet='light',epochs=300,shard=8,alpha=0.9"
    "exp3:use_resnet=False,split_alexnet='light',epochs=300,shard=2,alpha=0.3"
)

# 총 실험 개수
total_exp=${#experiments[@]}

# 각 실험 실행
for i in "${!experiments[@]}"; do
    exp_num=$((i+1))
    exp_data="${experiments[$i]}"
    exp_name=$(echo "$exp_data" | cut -d':' -f1)
    exp_params=$(echo "$exp_data" | cut -d':' -f2)
    
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  Experiment $exp_num/$total_exp: $exp_name${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo "Parameters: $exp_params"
    echo ""
    
    # GAS_main.py 복원
    echo -e "${GREEN}[2/4] Restoring GAS_main.py from backup...${NC}"
    cp GAS_main.py.backup GAS_main.py
    
    # 파라미터 수정
    echo -e "${GREEN}[3/4] Applying experiment parameters...${NC}"
    IFS=',' read -ra PARAMS <<< "$exp_params"
    for param in "${PARAMS[@]}"; do
        key=$(echo "$param" | cut -d'=' -f1)
        value=$(echo "$param" | cut -d'=' -f2-)
        
        echo "  Setting $key = $value"
        
        # sed로 파라미터 값 변경
        sed -i "s/^$key = .*/$key = $value/" GAS_main.py
    done
    echo ""
    
    # 실험 실행
    echo -e "${GREEN}[4/4] Running experiment...${NC}"
    echo "Command: python GAS_main.py"
    echo "Start time: $(date)"
    echo ""
    
    python GAS_main.py
    exit_code=$?
    
    echo ""
    echo "End time: $(date)"
    
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}✓ Experiment $exp_name completed successfully${NC}"
    else
        echo -e "${RED}✗ Experiment $exp_name failed with exit code $exit_code${NC}"
    fi
    
    # 결과 파일 백업
    if [ -f "GAS_main.txt" ]; then
        mv GAS_main.txt "results_${exp_name}.txt"
        echo "Results saved to: results_${exp_name}.txt"
    fi
    
    echo ""
    
    # 마지막 실험이 아니면 대기
    if [ $exp_num -lt $total_exp ]; then
        echo "Waiting 5 seconds before next experiment..."
        sleep 5
        echo ""
    fi
done

# 원본 복원
echo -e "${GREEN}Restoring original GAS_main.py...${NC}"
cp GAS_main.py.backup GAS_main.py

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  All experiments completed!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Results summary:"
ls -lh results_exp*.txt 2>/dev/null || echo "No result files found"
