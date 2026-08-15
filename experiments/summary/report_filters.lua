-- =====================================================================
-- report_filters.lua — JSSP 技术报告 pandoc Lua 过滤器（仅排版，不改内容）
--
--  1. 剥离标题中的手写章节编号（"1. " / "4.1 "），交给 LaTeX 自动编号
--  2. "摘要" 章节 → tcolorbox 摘要框（含"关键词"段）
--  3. 删除手写"目录"章节（由 pandoc --toc 自动目录替代：点线引导 + 页码）
--  4. "参考文献" → 无编号章节并写入目录
--  5. 所有表格 → booktabs 三线表 + tabularx（按内容自适应列宽）+ 深蓝表头
--  6. 所有代码块 → tcolorbox 浅灰底 + 边框（fancyvrb Verbatim）
--
-- 配套：report_template.tex / build_report.sh
-- =====================================================================

local latex_escapes = {
  ['\\'] = '\\textbackslash{}',
  ['&']  = '\\&',
  ['%']  = '\\%',
  ['$']  = '\\$',
  ['#']  = '\\#',
  ['_']  = '\\_',
  ['{']  = '\\{',
  ['}']  = '\\}',
  ['~']  = '\\textasciitilde{}',
  ['^']  = '\\textasciicircum{}',
}

local function tex_escape(s)
  return (s:gsub('.', function(c) return latex_escapes[c] or c end))
end

-- ----------------------------------------------------------------
-- 引号方向修复（P1-3）：pandoc smart 在中文语境下把 "…" 两侧都判为
-- 右引号 U+201D（如 `模型"多输出"` → 模型”多输出”）。这里按块内
-- 成对出现的 “/”/"（U+201C / U+201D / ASCII "）顺序转换为中文
-- 书名式引号 「…」，状态按块（段落/表格/标题）重置，避免跨块串对。
-- ----------------------------------------------------------------
local quote_open = false

-- 按 UTF-8 字节序列匹配（s:sub(i,i) 只取 1 字节，不能与 3 字节字面量比较）
local function match_utf8(s, i, bytes)
  local n = #bytes
  for j = 1, n do
    if s:byte(i + j - 1) ~= bytes[j] then return false end
  end
  return true
end

local LQUOTE = { 226, 128, 156 }   -- “
local RQUOTE = { 226, 128, 157 }   -- ”
local CN_LQ = '\227\128\140'       -- 「 U+300C
local CN_RQ = '\227\128\141'       -- 」 U+300D（注意是 8D，不是 91＝】）

local function fix_quotes(s)
  local out = {}
  local i = 1
  while i <= #s do
    local b = s:byte(i)
    if b == 34 then               -- ASCII "
      if quote_open then
        out[#out + 1] = CN_RQ; quote_open = false
      else
        out[#out + 1] = CN_LQ; quote_open = true
      end
      i = i + 1
    elseif match_utf8(s, i, LQUOTE) or match_utf8(s, i, RQUOTE) then
      if quote_open then
        out[#out + 1] = CN_RQ; quote_open = false
      else
        out[#out + 1] = CN_LQ; quote_open = true
      end
      i = i + 3
    else
      out[#out + 1] = s:sub(i, i)
      i = i + 1
    end
  end
  return table.concat(out)
end

-- 块级重置（正文 Str 在全局 Str 过滤器里做转换）
local function reset_quote_state()
  quote_open = false
end

-- 行内元素 → LaTeX 字符串
local function inlines(blocks)
  local parts = {}
  local function walk(blks)
    for _, b in ipairs(blks) do
      if b.t == 'Str' then
        parts[#parts + 1] = tex_escape(fix_quotes(b.text))
      elseif b.t == 'Space' or b.t == 'SoftBreak' then
        parts[#parts + 1] = ' '
      elseif b.t == 'LineBreak' then
        parts[#parts + 1] = '\\\\'
      elseif b.t == 'Strong' then
        parts[#parts + 1] = '\\textbf{' .. inlines(b.content) .. '}'
      elseif b.t == 'Emph' then
        parts[#parts + 1] = '\\emph{' .. inlines(b.content) .. '}'
      elseif b.t == 'Code' then
        parts[#parts + 1] = '\\texttt{' .. tex_escape(b.text) .. '}'
      elseif b.t == 'Math' then
        parts[#parts + 1] = '$' .. b.text .. '$'
      elseif b.t == 'RawInline' and b.format == 'tex' then
        parts[#parts + 1] = b.text
      elseif b.t == 'Link' then
        parts[#parts + 1] = inlines(b.content)
      else
        parts[#parts + 1] = tex_escape(pandoc.utils.stringify(b))
      end
    end
  end
  walk(blocks)
  return table.concat(parts)
end

local function cell_text(cell)
  return pandoc.utils.stringify(cell.contents)
end

local function cell_latex(cell)
  local parts = {}
  for _, blk in ipairs(cell.contents) do
    if blk.t == 'Para' or blk.t == 'Plain' then
      parts[#parts + 1] = inlines(blk.content)
    else
      parts[#parts + 1] = tex_escape(pandoc.utils.stringify(blk))
    end
  end
  -- "/" 后加断点：长 token（如 6×6/10×10、train/val/test）在窄列中
  -- 可从 "/" 后断行，避免整段不可断造成 Overfull。
  -- 注意（本机 XeTeX 实测的坑）：
  --   ① 不用 \allowbreak——鲁棒命令后紧跟 UTF-8 CJK 字节报 Undefined cs；
  --   ② 不用裸 \penalty0——后跟数字会被并入数值（\penalty010 吃掉 "10"）；
  --   ③ 不用 \z@——正文中 @ 是 other catcode，\z@ 被拆成 \z + @；
  --   ④ 控制字后紧跟 CJK 字节同样报 Undefined cs，故 \relax 后必须留空格
  --      （控制字后的空格会被吞掉，不产生可见间隙）。
  -- 用 \penalty0\relax （尾随空格）：\relax 终止数字扫描且吞掉空格。
  return (table.concat(parts, '\\par ')):gsub('/', '/\\penalty0\\relax ')
end

-- ---------------- 1-4. 标题处理 ----------------
local in_abstract = false
local remove_next_list = false -- 手写目录的链接列表（"## 目录"之后首个列表）

local function strip_number(plain)
  local rest = plain:match('^%d+%.[%d%.]*%s+(.*)$')
  if not rest then rest = plain:match('^%d+%s+(.*)$') end
  return rest
end

function Header(el)
  if el.level == 2 then
    local text = pandoc.utils.stringify(el.content)
    if text == '摘要' then
      in_abstract = true
      return { pandoc.RawBlock('latex', '\\begin{abstractbox}') }
    elseif text == '目录' then
      local out = {}
      if in_abstract then
        in_abstract = false
        out[#out + 1] = pandoc.RawBlock('latex', '\\end{abstractbox}')
      end
      remove_next_list = true -- 手写目录标题后的链接列表一并删除
      return out
    elseif text == '参考文献' then
      local out = {}
      if in_abstract then
        in_abstract = false
        out[#out + 1] = pandoc.RawBlock('latex', '\\end{abstractbox}')
      end
      out[#out + 1] = pandoc.RawBlock('latex',
        '\\section*{参考文献}\\addcontentsline{toc}{section}{参考文献}')
      return out
    else
      local out = {}
      if in_abstract then
        in_abstract = false
        out[#out + 1] = pandoc.RawBlock('latex', '\\end{abstractbox}')
      end
      local rest = strip_number(text)
      if rest then el.content = pandoc.List({ pandoc.Str(rest) }) end
      el.level = el.level - 1 -- H1 已剥离，## 从 \section 起算
      out[#out + 1] = el
      return out
    end
  else
    local text = pandoc.utils.stringify(el.content)
    local rest = strip_number(text)
    if rest then el.content = pandoc.List({ pandoc.Str(rest) }) end
    el.level = el.level - 1 -- ## 已在上一层降级为 \section，### 相应为 \subsection
    return el
  end
end

-- 删除手写目录的链接列表（紧跟 "## 目录" 的 OrderedList）
function OrderedList(el)
  if remove_next_list then
    remove_next_list = false
    return {}
  end
  return nil
end

-- ---------------- 引号修复：正文 Str 全局转换 ----------------
-- 块级入口重置配对状态（块内成对转换；"每段落独立配对"避免跨块串对）
function Para(el)
  reset_quote_state()
  return nil
end
function Plain(el)
  reset_quote_state()
  return nil
end
function Str(el)
  el.text = fix_quotes(el.text)
  return el
end

-- ---------------- 5. 表格 ----------------
function Table(tbl)
  -- 表格单元格引号独立配对
  reset_quote_state()
  -- pandoc 3.1.3 AST：head.rows[1].cells = 表头；body.body = 数据行
  local header_cells = {}
  if tbl.head and tbl.head.rows and #tbl.head.rows > 0 then
    for _, c in ipairs(tbl.head.rows[1].cells) do
      header_cells[#header_cells + 1] = c
    end
  end
  local body_rows = {}
  for _, body in ipairs(tbl.bodies) do
    for _, row in ipairs(body.head) do body_rows[#body_rows + 1] = row.cells end
    for _, row in ipairs(body.body) do body_rows[#body_rows + 1] = row.cells end
  end
  local ncols = #header_cells
  if ncols == 0 then return nil end
  -- 复杂单元格（合并等）→ 退回默认渲染
  for _, row in ipairs(body_rows) do
    for _, c in ipairs(row) do
      if (c.row_span or 1) ~= 1 or (c.col_span or 1) ~= 1 then return nil end
    end
  end

  -- 自适应列宽：按各列最长内容"排版宽度"分配 \hsize。
  -- CJK 字符（全角）计 2 个单位，ASCII 计 1 个单位——比字符数更接近
  -- 真实列宽比例，缓解 §2.4 这类长中文列被低估导致的 Overfull。
  -- 注意：必须按 UTF-8 码点遍历（Lua 的 "." 匹配字节，3 字节 CJK
  -- 会被数成 3 个"字符"）。
  local function text_units(s)
    local units = 0
    for _, cp in utf8.codes(s) do
      if cp > 127 then units = units + 2 else units = units + 1 end
    end
    return units
  end
  local weights = {}
  for i = 1, ncols do weights[i] = 4 end
  local function consider(cells)
    for i, c in ipairs(cells) do
      local w = text_units(cell_text(c))
      if w > weights[i] then weights[i] = w end
    end
  end
  consider(header_cells)
  for _, row in ipairs(body_rows) do consider(row) end
  local total = 0
  for i = 1, ncols do total = total + weights[i] end
  local hsizes = {}
  for i = 1, ncols do
    hsizes[i] = math.max(0.5, weights[i] / total * ncols)
  end
  local hsum = 0
  for i = 1, ncols do hsum = hsum + hsizes[i] end
  for i = 1, ncols do hsizes[i] = hsizes[i] / hsum * ncols end

  -- 列宽：p{}-列精确宽度（不用 tabularx——其测量遍会打印伪警告
  -- "Underfull badness 10000 in alignment"，且 X 列宽度不可精确控制）。
  -- W_i = hsize_i/ncols·\linewidth − f·\tabcolsep，f = 2(n−1)/n，
  -- 使整表宽度恰为 \linewidth（@{} 去掉两端 colsep，列间各 2\tabcolsep）。
  -- 单元格用 \RaggedRight（ragged2e，右留白 1fil 伸缩）避免断行 underfull。
  local tcf = 2 * (ncols - 1) / ncols
  local cols = {}
  for i = 1, ncols do
    local spec = tbl.colspecs and tbl.colspecs[i]
    local align = spec and (spec[1] or spec.alignment) or 'AlignDefault'
    local ralign
    if align == 'AlignRight' then
      ralign = 'RaggedLeft'
    elseif align == 'AlignCenter' then
      ralign = 'centering'
    else
      ralign = 'RaggedRight'
    end
    cols[i] = string.format(
      '>{\\%s\\arraybackslash}p{\\dimexpr%0.4f\\linewidth/%d-%0.3f\\tabcolsep\\relax}',
      ralign, hsizes[i], ncols, tcf)
  end

  local rows_tex = {}
  local hcells = {}
  for _, c in ipairs(header_cells) do
    hcells[#hcells + 1] = '{\\sffamily\\bfseries\\color{white} '
      .. cell_latex(c) .. '}'
  end
  rows_tex[#rows_tex + 1] = '\\toprule\n\\rowcolor{dblue}\n'
    .. table.concat(hcells, ' & ') .. ' \\\\\n\\midrule'
  for _, row in ipairs(body_rows) do
    local cells = {}
    for _, c in ipairs(row) do cells[#cells + 1] = cell_latex(c) end
    rows_tex[#rows_tex + 1] = table.concat(cells, ' & ') .. ' \\\\'
  end
  rows_tex[#rows_tex + 1] = '\\bottomrule'

  local tex = '\\begin{table}[H]\n\\centering\n'
    .. '\\renewcommand{\\arraystretch}{1.3}\n'
    .. '\\begin{tabular}{@{}' .. table.concat(cols, ' ') .. '@{}}\n'
    .. table.concat(rows_tex, '\n') .. '\n\\end{tabular}\n\\end{table}'
  return pandoc.RawBlock('latex', tex)
end

-- ---------------- 6. 代码块 ----------------
function CodeBlock(el)
  if el.text:find('\\end{Verbatim}') then return nil end
  local tex = '\\begin{codebox}\n\\begin{Verbatim}[fontsize=\\small]\n'
    .. el.text .. '\n\\end{Verbatim}\n\\end{codebox}'
  return pandoc.RawBlock('latex', tex)
end
