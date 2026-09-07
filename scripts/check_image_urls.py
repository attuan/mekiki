"""Craigslist の image_url がまだ生きているかを少数サンプルで確認する。

PRD の P5（画像モダリティ）に着手してよいかの事前判定用。
6万件をダウンロードする前に、100件で生存率と1件あたりの所要時間を測り、
全件の見積もりを出す。データは2021年収集なので、URL が失効している可能性がある。

使い方:
    .venv/bin/python scripts/check_image_urls.py            # 100件
    .venv/bin/python scripts/check_image_urls.py -n 300     # 件数を変える

判定の注意: craigslist は削除済み投稿でも HTTP 200 を返し、
「画像なし」のプレースホルダ画像を配ることがある。ステータスコードだけでは
生存を判定できないので、本文のハッシュを取って同一画像の多発を検出する。
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import io
import statistics
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import duckdb
import requests

RAW = "sampledata/raw/vehicles.csv"
CLEAN = "sampledata/processed/vehicles_multi_clean.parquet"
TIMEOUT = 15
WORKERS = 8
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "\
     "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"


def sample_urls(n: int, seed: int = 42) -> list[tuple[int, str]]:
    """クレンジング後の行に対応する image_url を n 件サンプリングする。"""
    con = duckdb.connect()
    rows = con.sql(f"""
        SELECT r.id, r.image_url
        FROM read_csv_auto('{RAW}', ignore_errors=true) r
        JOIN read_parquet('{CLEAN}') c ON CAST(r.id AS VARCHAR) = CAST(c.id AS VARCHAR)
        WHERE r.image_url IS NOT NULL AND r.image_url <> ''
        USING SAMPLE {n} ROWS (reservoir, {seed})
    """).fetchall()
    return [(int(i), u) for i, u in rows]


def jpeg_size(data: bytes) -> tuple[int, int] | None:
    """JPEG のヘッダから (幅, 高さ) を読む。Pillow を足さずに済ませるため。"""
    f = io.BytesIO(data)
    if f.read(2) != b"\xff\xd8":
        return None
    while True:
        b = f.read(1)
        while b and b != b"\xff":
            b = f.read(1)
        marker = f.read(1)
        if not marker:
            return None
        m = marker[0]
        if m in (0xD8, 0x01) or 0xD0 <= m <= 0xD7:
            continue
        seg = f.read(2)
        if len(seg) < 2:
            return None
        length = struct.unpack(">H", seg)[0]
        if 0xC0 <= m <= 0xCF and m not in (0xC4, 0xC8, 0xCC):
            body = f.read(7)
            if len(body) < 5:
                return None
            h, w = struct.unpack(">HH", body[1:5])
            return w, h
        f.seek(length - 2, 1)


def fetch(item: tuple[int, str]) -> dict:
    vid, url = item
    t0 = time.time()
    try:
        r = requests.get(url, timeout=TIMEOUT, headers={"User-Agent": UA})
        data = r.content
        return {
            "id": vid, "url": url, "status": r.status_code,
            "bytes": len(data), "sec": time.time() - t0,
            "ctype": r.headers.get("Content-Type", ""),
            "sha1": hashlib.sha1(data).hexdigest()[:12],
            "wh": jpeg_size(data),
            "error": None,
        }
    except Exception as e:  # ネットワーク断・タイムアウト等
        return {"id": vid, "url": url, "status": None, "bytes": 0,
                "sec": time.time() - t0, "ctype": "", "sha1": "",
                "wh": None, "error": type(e).__name__}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=100, help="確認する件数")
    ap.add_argument("--workers", type=int, default=WORKERS, help="並列数")
    args = ap.parse_args()

    print(f"サンプリング中（{args.n}件）...", flush=True)
    urls = sample_urls(args.n)
    print(f"取得対象 {len(urls)} 件 / 並列 {args.workers}\n", flush=True)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(fetch, urls))
    wall = time.time() - t0

    ok = [r for r in results if r["status"] == 200 and r["bytes"] > 0]
    errs = [r for r in results if r["error"]]
    bad = [r for r in results if r["status"] not in (200, None)]

    # プレースホルダ検出: 同じ画像が何度も返っていないか
    dup = collections.Counter(r["sha1"] for r in ok)
    placeholders = [(h, c) for h, c in dup.most_common(3) if c > 1]

    print("=" * 62)
    print(f"HTTP 200 かつ本文あり : {len(ok):>4} / {len(results)}  "
          f"({len(ok) / len(results) * 100:.1f}%)")
    print(f"HTTP エラー          : {len(bad):>4}  "
          f"{collections.Counter(r['status'] for r in bad).most_common()}")
    print(f"通信失敗             : {len(errs):>4}  "
          f"{collections.Counter(r['error'] for r in errs).most_common()}")
    if ok:
        sizes = sorted(r["bytes"] for r in ok)
        secs = sorted(r["sec"] for r in ok)
        whs = collections.Counter(str(r["wh"]) for r in ok)
        print(f"\n画像サイズ 中央値    : {statistics.median(sizes)/1024:.0f} KB "
              f"(最小 {sizes[0]/1024:.0f} / 最大 {sizes[-1]/1024:.0f})")
        print(f"1件の応答 中央値     : {statistics.median(secs):.2f} 秒")
        print(f"解像度               : {whs.most_common(3)}")
    if placeholders:
        print(f"\n⚠ 同一画像の重複      : {placeholders}")
        print("  → 同じ画像が複数URLで返っている。プレースホルダの疑い")
    else:
        print("\n同一画像の重複        : なし（全て異なる画像）")

    print("\n" + "-" * 62)
    print(f"実測: {len(results)}件を並列{args.workers}で {wall:.1f} 秒 "
          f"= {wall/len(results)*1000:.0f} ms/件")
    if ok:
        n_full = 200_374
        est_sec = wall / len(results) * n_full
        est_gb = statistics.median(r["bytes"] for r in ok) * n_full * len(ok) / len(results) / 1e9
        print(f"全件({n_full:,}行)の見積もり: 同じ並列数で {est_sec/3600:.1f} 時間 / "
              f"約 {est_gb:.1f} GB")
        print(f"  6万行サブセットなら       : {est_sec*60000/n_full/3600:.1f} 時間")
        print(f"  並列を32に上げれば        : "
              f"{est_sec*60000/n_full/3600*args.workers/32:.1f} 時間（6万行）")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
