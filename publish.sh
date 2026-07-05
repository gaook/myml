#!/usr/bin/env bash
# 一键发布：本地构建自检 → 提交 → 推送到 GitHub
# 推送后 Cloudflare Pages 会自动构建上线（约 1 分钟）
# 用法：./publish.sh "这次更新的说明"
set -e
cd "$(dirname "$0")"

MSG="${1:-update: $(date '+%Y-%m-%d %H:%M')}"

echo "▶ 本地构建自检（不生成文件，只检查有没有报错）..."
hugo --gc --minify --renderToMemory

echo "▶ 提交改动：$MSG"
git add -A
if git diff --cached --quiet; then
  echo "✔ 没有需要提交的改动，退出。"
  exit 0
fi
git commit -m "$MSG"

echo "▶ 推送到 GitHub..."
git push

echo "✅ 已推送。Cloudflare Pages 正在自动构建，约 1 分钟后上线。"
