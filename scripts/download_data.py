#!/usr/bin/env python3
"""Kaggle の Craigslist 中古車データを sampledata/raw/ に取得する。

このリポジトリには 1.4GB の生データを含めていないため、
clone した直後は各自でこのスクリプトを実行する必要がある。

    python3 scripts/download_data.py

事前準備（初回のみ）:
  仮想環境を有効化した上で、次のどれかで Kaggle 認証を通す。

  A) .env に書く（このリポジトリの流儀。ANTHROPIC_API_KEY と同じ置き場所）
       https://www.kaggle.com/settings の "API" で Generate New Token を押し、
       KAGGLE_API_TOKEN= の右に貼る
  B) ブラウザ認証（トークンの管理が要らない）
       kaggle auth login
  C) 環境変数またはファイル
       export KAGGLE_API_TOKEN=xxxxx  もしくは ~/.kaggle/access_token に保存

kaggle CLI 自身は .env を見ない（環境変数・~/.kaggle・ブラウザ認証しか探さない）。
A) を成立させるため、このスクリプトが .env を読んで子プロセスの環境に渡す。
"""
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

DATASET = "austinreese/craigslist-carstrucks-data"
RAW_DIR = Path(__file__).resolve().parent.parent / "sampledata" / "raw"
TARGET = RAW_DIR / "vehicles.csv"
EXPECTED_BYTES = 1_447_955_215  # 2021-05-06 版のサイズ
ENV_PATH = RAW_DIR.parent.parent / ".env"

#: kaggle CLI が資格情報として読む環境変数（kagglesdk/kaggle_env.py の実装順）。
#: KAGGLE_API_TOKEN が最優先で、無ければ USERNAME + KEY の組が使われる。
KAGGLE_ENV_KEYS = ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY")
#: ~/.kaggle に置かれていれば CLI が自力で見つけるファイル。
KAGGLE_CRED_FILES = ("access_token", "access_token.txt", "kaggle.json")


def read_dotenv(path: Path) -> dict:
    """.env を素朴に読む。python-dotenv が入っていない環境でも動かすため。

    書式は unfold/llm.py の load_api_key() と同じ（KEY=VALUE、# はコメント）。
    値は決して表示しない。キーが標準出力やログに残るのを避けるため。
    """
    if not path.exists():
        return {}
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        value = value.strip().strip('"').strip("'")
        if value:
            values[name.strip()] = value
    return values


def kaggle_env() -> dict:
    """kaggle CLI に渡す環境を組み立てる。既存の環境変数を .env より優先する。

    逆にすると、一時的に別アカウントで叩きたくて export した値が
    .env に上書きされてしまう。
    """
    env = os.environ.copy()
    from_file = read_dotenv(ENV_PATH)
    injected = [key for key in KAGGLE_ENV_KEYS
                if not env.get(key) and from_file.get(key)]
    for key in injected:
        env[key] = from_file[key]
    if injected:
        print(f"認証情報を .env から読みました: {', '.join(injected)}")
    return env


def has_credentials(env: dict) -> bool:
    """CLI が認証に使えるものが1つでもあるか。無ければ叩く前に止める。"""
    if any(env.get(key) for key in KAGGLE_ENV_KEYS):
        return True
    return any((Path.home() / ".kaggle" / name).exists()
               for name in KAGGLE_CRED_FILES)


def report_no_credentials() -> None:
    """どこにも資格情報が無いとき、どこを直せばよいかを示す。"""
    print(
        "Kaggle の認証情報が見つかりません。次のどれも設定されていません:\n"
        f"  - {ENV_PATH} の KAGGLE_API_TOKEN\n"
        "  - 環境変数 KAGGLE_API_TOKEN（または KAGGLE_USERNAME と KAGGLE_KEY）\n"
        "  - ~/.kaggle/access_token, ~/.kaggle/kaggle.json\n"
        "  - ブラウザ認証（kaggle auth login）\n"
        "\n"
        "https://www.kaggle.com/settings の API から Generate New Token で発行し、\n"
        f"{ENV_PATH} の KAGGLE_API_TOKEN= の右に貼ってください。",
        file=sys.stderr,
    )


def already_present() -> bool:
    if not TARGET.exists():
        return False
    size = TARGET.stat().st_size
    if size == EXPECTED_BYTES:
        print(f"取得済み: {TARGET} ({size:,} bytes)")
        return True
    print(
        f"警告: {TARGET} は存在しますがサイズが想定と異なります。\n"
        f"  実際  : {size:,} bytes\n"
        f"  想定  : {EXPECTED_BYTES:,} bytes\n"
        "  データセットが更新された可能性があります。再取得する場合は"
        "このファイルを削除してから再実行してください。"
    )
    return True


def main() -> int:
    if already_present():
        return 0

    if shutil.which("kaggle") is None:
        print(
            "kaggle コマンドが見つかりません。仮想環境を有効化してください:\n"
            "  source .venv/bin/activate\n"
            "未セットアップなら: python3.12 -m venv .venv && "
            ".venv/bin/pip install -r requirements.txt\n"
            "認証手順はこのファイル冒頭のコメントを参照。\n"
            "\n"
            "手動で取得する場合は次の URL から vehicles.csv をダウンロードし、\n"
            f"  https://www.kaggle.com/datasets/{DATASET}\n"
            f"{RAW_DIR} に置いてください。",
            file=sys.stderr,
        )
        return 1

    env = kaggle_env()
    if not has_credentials(env):
        report_no_credentials()
        return 1

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Kaggle から取得中: {DATASET}（1.4GB あるため数分かかります）")
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", DATASET, "-p", str(RAW_DIR)],
        env=env,
    )
    if result.returncode != 0:
        print("kaggle コマンドが失敗しました。認証設定を確認してください。", file=sys.stderr)
        return result.returncode

    archives = list(RAW_DIR.glob("*.zip"))
    if not archives:
        print("zip が見つかりません。取得結果を確認してください。", file=sys.stderr)
        return 1

    for archive in archives:
        print(f"展開中: {archive.name}")
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(RAW_DIR)
        archive.unlink()

    if not TARGET.exists():
        print(f"展開後も {TARGET} が見つかりません。", file=sys.stderr)
        return 1

    print(f"完了: {TARGET} ({TARGET.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
