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

-- 行内元素 → LaTeX 字符串
local function inlines(blocks)
  local parts = {}
  local function walk(blks)
    for _, b in ipairs(blks) do
      if b.t == 'Str' then
        parts[#parts + 1] = tex_escape(b.text)
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
  return table.concat(parts, '\\par ')
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

-- ---------------- 5. 表格 ----------------
function Table(tbl)
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

  -- 自适应列宽：按各列最长内容长度分配 \hsize
  local weights = {}
  for i = 1, ncols do weights[i] = 4 end
  local function consider(cells)
    for i, c in ipairs(cells) do
      local w = #cell_text(c)
      if w > weights[i] then weights[i] = w end
    end
  end
  consider(header_cells)
  for _, row in ipairs(body_rows) do consider(row) end
  local total = 0
  for i = 1, ncols do total = total + weights[i] end
  local hsizes = {}
  for i = 1, ncols do
    hsizes[i] = math.max(0.55, weights[i] / total * ncols)
  end
  local hsum = 0
  for i = 1, ncols do hsum = hsum + hsizes[i] end
  for i = 1, ncols do hsizes[i] = hsizes[i] / hsum * ncols end

  local cols = {}
  for i = 1, ncols do
    local spec = tbl.colspecs and tbl.colspecs[i]
    local align = spec and (spec[1] or spec.alignment) or 'AlignDefault'
    local ralign
    if align == 'AlignRight' then
      ralign = 'raggedleft'
    elseif align == 'AlignCenter' then
      ralign = 'centering'
    else
      ralign = 'raggedright'
    end
    cols[i] = string.format(
      '>{\\hsize=%0.3f\\hsize\\%s\\arraybackslash}X', hsizes[i], ralign)
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
    .. '\\begin{tabularx}{\\linewidth}{@{}' .. table.concat(cols, ' ') .. '@{}}\n'
    .. table.concat(rows_tex, '\n') .. '\n\\end{tabularx}\n\\end{table}'
  return pandoc.RawBlock('latex', tex)
end

-- ---------------- 6. 代码块 ----------------
function CodeBlock(el)
  if el.text:find('\\end{Verbatim}') then return nil end
  local tex = '\\begin{codebox}\n\\begin{Verbatim}[fontsize=\\small]\n'
    .. el.text .. '\n\\end{Verbatim}\n\\end{codebox}'
  return pandoc.RawBlock('latex', tex)
end
