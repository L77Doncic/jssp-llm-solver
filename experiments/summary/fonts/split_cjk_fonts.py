#!/usr/bin/env python3
# =====================================================================
# split_cjk_fonts.py — 将系统 Noto CJK TTC 按 SC 面拆分为单面 OTF
#
# 背景：TeX Live 2022/2023 的 fontspec 2.8a 无法选择 TTC 内 face index，
# 按字体名加载 "Noto Serif CJK SC" 恒取 face 0（JP 面），且该 TTC 各面
# 共享 CFF（FontName 均为 *jp*），导致 pdffonts 全部显示 *CJKjp*。
# 本脚本按 face index 拆出 SC 面（含 Mono 面），并把 CFF FontName 改为
# *sc*，使 XeTeX 嵌入的字体名与 SC 声明一致。
#
# 用法：python3 split_cjk_fonts.py [输出目录]
#   默认输出目录：本脚本所在目录（fonts/）
# 依赖：pip install fonttools
# =====================================================================
import os
import shutil
import sys

from fontTools.ttLib import TTCollection

# 楷体（斜体强调用，中文字体惯例）：系统字体拷入 fonts/ 供 Path= 加载
KAITI = '/usr/share/fonts/truetype/arphic-gkai00mp/gkai00mp.ttf'

# (源 TTC, face index, 输出文件名, CFF FontName)
JOBS = [
    ('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
     2, 'NotoSansCJKsc-Regular.otf', 'NotoSansCJKsc-Regular'),
    ('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc',
     2, 'NotoSansCJKsc-Bold.otf', 'NotoSansCJKsc-Bold'),
    ('/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc',
     2, 'NotoSerifCJKsc-Regular.otf', 'NotoSerifCJKsc-Regular'),
    ('/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc',
     2, 'NotoSerifCJKsc-Bold.otf', 'NotoSerifCJKsc-Bold'),
    ('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
     7, 'NotoSansMonoCJKsc-Regular.otf', 'NotoSansMonoCJKsc-Regular'),
    ('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc',
     7, 'NotoSansMonoCJKsc-Bold.otf', 'NotoSansMonoCJKsc-Bold'),
]


def main() -> int:
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(KAITI):
        shutil.copy2(KAITI, os.path.join(out_dir, 'ARPLKaitiMGB.ttf'))
        print(f'{out_dir}/ARPLKaitiMGB.ttf: AR PL KaitiM GB (楷体, 斜体用)')
    for path, index, out_name, cff_name in JOBS:
        tc = TTCollection(path)
        font = tc.fonts[index]
        real_name = font['name'].getDebugName(4)
        assert 'SC' in real_name, f'face {index} of {path} is not SC: {real_name}'
        font['CFF '].cff.fontNames = [cff_name]
        out = os.path.join(out_dir, out_name)
        font.save(out)
        print(f'{out}: {real_name}  (CFF FontName -> {cff_name})')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
