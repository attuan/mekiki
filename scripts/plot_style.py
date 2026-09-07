"""グラフの日本語フォントを、動いている OS に合わせて選ぶ。

macOS には Hiragino Sans が、Ubuntu には Noto Sans CJK JP が入っている。
どちらか一方を決め打ちすると、**もう片方の環境で日本語が豆腐（□）になる**。
計算ノード（Ubuntu 24.04）へ移行するにあたり、実際にインストールされている
フォントを調べて使う形にした。

使い方は、作図する前に一度呼ぶだけ:

    from plot_style import use_japanese_font
    use_japanese_font()

見つからなければ警告を出し、入れ方を案内する（グラフ自体は豆腐で描かれる）。
"""

from __future__ import annotations

import platform
import warnings

import matplotlib.pyplot as plt
from matplotlib import font_manager

# 上から順に探し、最初に見つかったものを使う。
CANDIDATES = (
    "Hiragino Sans",              # macOS（標準搭載）
    "Hiragino Kaku Gothic ProN",  # macOS の古い版
    "Noto Sans CJK JP",           # Ubuntu: sudo apt install fonts-noto-cjk
    "Noto Sans JP",
    "IPAexGothic",                # Ubuntu: sudo apt install fonts-ipaexfont
    "TakaoPGothic",
    "Yu Gothic",                  # Windows
    "MS Gothic",
)

_HOWTO = {
    "Linux": "sudo apt install fonts-noto-cjk",
    "Darwin": "Hiragino Sans は標準で入っているはずです。OS の状態を確認してください",
    "Windows": "Yu Gothic は標準で入っているはずです",
}


def pick_japanese_font() -> str | None:
    """インストール済みの日本語フォント名を1つ返す。無ければ None。"""
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in CANDIDATES:
        if name in installed:
            return name
    return None


def use_japanese_font(verbose: bool = False) -> str | None:
    """日本語フォントを rcParams に設定する。使った名前を返す。

    `axes.unicode_minus` はフォントの有無にかかわらず False にする
    （True のままだと負の目盛りのマイナス記号が豆腐になる）。
    """
    plt.rcParams["axes.unicode_minus"] = False

    name = pick_japanese_font()
    if name is None:
        howto = _HOWTO.get(platform.system(), "日本語フォントを導入してください")
        warnings.warn(
            "日本語フォントが見つかりません。グラフの日本語が □ になります。\n"
            f"  導入方法: {howto}\n"
            f"  探した候補: {', '.join(CANDIDATES)}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    plt.rcParams["font.family"] = name
    if verbose:
        print(f"日本語フォント: {name}")
    return name
