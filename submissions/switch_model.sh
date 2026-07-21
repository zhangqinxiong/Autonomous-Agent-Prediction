#!/bin/bash
# Usage: ./switch_model.sh local    # 切换到 deepseek-v3.2（本地测试）
#        ./switch_model.sh kaggle   # 切换到 gemini-3.5-flash（提交Kaggle）
#        ./switch_model.sh          # 显示当前使用的模型

AGENT_DIR="$(cd "$(dirname "$0")/.." && pwd)/agent"
YAML="$AGENT_DIR/agent.yaml"

if [ ! -f "$YAML" ]; then
    echo "Error: agent.yaml not found at $YAML"
    exit 1
fi

CURRENT=$(grep "^model:" "$YAML" | sed 's/model: //')

if [ "$1" = "local" ]; then
    sed -i 's/^model:.*/model: deepseek-v3.2/' "$YAML"
    echo "Switched to deepseek-v3.2 (local test)"
elif [ "$1" = "kaggle" ]; then
    sed -i 's/^model:.*/model: gemini-3.5-flash/' "$YAML"
    echo "Switched to gemini-3.5-flash (Kaggle submission)"
elif [ "$1" = "" ]; then
    echo "Current model: $CURRENT"
    echo ""
    echo "Usage: ./switch_model.sh local|kaggle"
else
    echo "Unknown: $1 (use local or kaggle)"
    exit 1
fi
