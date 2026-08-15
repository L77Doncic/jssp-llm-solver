#!/usr/bin/env bash
# =====================================================================
# build_report.sh — JSSP 技术报告 PDF 构建（pandoc + xelatex + 自定义模板）
#
# 用法：bash build_report.sh
# 产物：technical_report.pdf（覆盖旧版）+ technical_report.tex（中间产物）
# 依赖：report_template.tex（排版模板）/ report_filters.lua（排版过滤器）
#
# 说明：正文内容零修改——md 首行 H1 由封面渲染（见 technical_report.md
# 首行注释），手写目录由自动目录（--toc）替代，均属排版层转换。
# =====================================================================
set -euo pipefail
cd "$(dirname "$0")"

MD=technical_report.md
TITLE="$(head -1 "$MD" | sed 's/^# //')"

# 封面标题手工断行（P3-7）：在"（JSSP）"前拆成两段，
# 模板中以 \\ 断行渲染（pandoc 会转义 metadata 里的反斜杠，故不能直接传 \\）
TITLE1="$(printf '%s' "$TITLE" | sed 's/（JSSP）.*//')"
TITLE2="$(printf '%s' "$TITLE" | sed 's/.*（JSSP）/（JSSP）/')"
[ "$TITLE1" = "$TITLE" ] && TITLE2=""   # 无（JSSP）标记则退化为单行

# 1) 剥离首行 H1（标题由 --metadata title 传入封面）
tail -n +2 "$MD" > technical_report.body.md

# 2) markdown → LaTeX（模板 + 过滤器 + 自动编号目录）
pandoc technical_report.body.md -f markdown+smart \
  --lua-filter=report_filters.lua \
  --template=report_template.tex \
  --toc --toc-depth=2 --number-sections \
  --metadata title="$TITLE" \
  --metadata covertitle1="$TITLE1" \
  --metadata covertitle2="$TITLE2" \
  --metadata subtitle="数据构建 → SFT → FOARL → 自动评估：全流程实现、问题归因与实验分析" \
  --metadata date="2026年8月15日" \
  --metadata model="Qwen2.5-7B-Instruct（LoRA 微调）" \
  --metadata compute="RTX 5090（32 GB）× 1" \
  --metadata linestretch="1.22" \
  -o technical_report.tex

# 3) xelatex 两遍（第二遍生成目录页码）
xelatex -interaction=nonstopmode -halt-on-error technical_report.tex >/dev/null 2>&1 \
  || { echo "[失败] xelatex 第 1 遍编译错误"; tail -60 technical_report.log; exit 1; }
xelatex -interaction=nonstopmode -halt-on-error technical_report.tex >/dev/null 2>&1 \
  || { echo "[失败] xelatex 第 2 遍编译错误"; tail -60 technical_report.log; exit 1; }

# 4) 验证：缺失字符 / 字体警告 / 页数
grep -nE "Missing character|does not contain requested" technical_report.log \
  && { echo "[警告] 上述缺失字符/字体警告，请检查"; }
echo "构建完成：$(pwd)/technical_report.pdf"
pdfinfo technical_report.pdf | grep -E "^(Pages|Page size)"

# 清理中间文件
rm -f technical_report.body.md technical_report.aux technical_report.out
echo "（technical_report.tex / .log 保留，便于排查）"
