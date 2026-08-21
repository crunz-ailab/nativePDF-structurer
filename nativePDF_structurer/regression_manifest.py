"""创建并校验解析输出的字节级清单。

模块为每个普通输出文件记录排序后的
相对路径、字节大小和SHA-256摘要，可发现文件缺失、额外文件、重命名或
内容变化，并且不绑定任何具体PDF回归语料。

Create and verify byte-exact manifests for parser outputs.

Records sorted relative paths, byte sizes, and SHA-256 digests for every
regular output file. The generic command detects missing, unexpected,
renamed, or modified artifacts without depending on any specific PDF corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


MANIFEST_SCHEMA_VERSION = 1


#@GPT 输出资源可能包含较大的PNG；分块读取避免为了计算hash一次把整个文件载入内存，同时保持纯标准库实现。
def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of one regular file.

    分块读取普通文件并返回其SHA-256摘要，避免一次性加载大型PNG或CSV。
    """

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


#@GPT 文件清单只使用POSIX相对路径并按路径排序，使macOS/Linux生成的JSON稳定，且不会把开发者本机绝对路径写入Git仓库。
def build_manifest(
    output_dir: Path,
    *,
    label: str = "",
    source_pdf_name: str = "",
    page_spec: str = "",
    baseline_checkpoint: str = "",
) -> dict[str, Any]:
    """Build a deterministic manifest for every regular output file.

    为输出目录中的全部普通文件构建确定性清单。路径使用POSIX相对形式并按
    字典序排列，记录文件大小和SHA-256摘要。
    """

    root = Path(output_dir).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"output directory does not exist: {root}")

    files = []
    for path in sorted(
        (
            candidate
            for candidate in root.rglob("*")
            if candidate.is_file() and not candidate.is_symlink()
        ),
        key=lambda candidate: candidate.relative_to(root).as_posix(),
    ):
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "label": label,
        "source_pdf_name": source_pdf_name,
        "page_spec": page_spec,
        "baseline_checkpoint": baseline_checkpoint,
        "file_count": len(files),
        "files": files,
    }


def write_manifest(
    output_dir: Path,
    manifest_path: Path,
    **metadata: str,
) -> dict[str, Any]:
    """Create a manifest and write it as stable UTF-8 JSON.

    创建输出清单，并以稳定的UTF-8 JSON格式写入指定路径。
    """

    manifest = build_manifest(output_dir, **metadata)
    destination = Path(manifest_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


#@GPT 校验时同时比较路径集合、大小和hash；只比较已有文件hash会漏掉候选被重命名、图片重复输出或资源意外消失。
def verify_manifest(
    output_dir: Path,
    manifest_path: Path,
) -> list[str]:
    """Return human-readable mismatches; an empty list means exact match.

    比较实际输出与清单中的路径、大小和SHA-256；返回可读差异列表，空列表
    表示字节级完全一致。
    """

    root = Path(output_dir).resolve()
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        return [
            "unsupported manifest schema: "
            f"{manifest.get('schema_version')!r}"
        ]

    expected = {
        str(record["path"]): record
        for record in manifest.get("files", [])
    }
    actual_paths = {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }

    errors = []
    for relative_path in sorted(expected.keys() - actual_paths.keys()):
        errors.append(f"missing: {relative_path}")
    for relative_path in sorted(actual_paths.keys() - expected.keys()):
        errors.append(f"unexpected: {relative_path}")
    for relative_path in sorted(expected.keys() & actual_paths.keys()):
        path = actual_paths[relative_path]
        record = expected[relative_path]
        actual_size = path.stat().st_size
        if actual_size != int(record["size"]):
            errors.append(
                f"size mismatch: {relative_path}: "
                f"expected {record['size']}, got {actual_size}"
            )
            continue
        actual_digest = sha256_file(path)
        if actual_digest != record["sha256"]:
            errors.append(f"sha256 mismatch: {relative_path}")
    return errors


def build_parser() -> argparse.ArgumentParser:
    """Create the generic regression-manifest command-line interface.

    创建通用回归清单CLI，支持 ``create`` 和 ``verify`` 两个子命令，不绑定
    任何具体PDF或页面集合。
    """

    parser = argparse.ArgumentParser(
        prog="python -m nativePDF_structurer.regression_manifest",
        description="Create or verify a byte-exact parser-output manifest.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create", help="write a new manifest")
    create.add_argument("output_dir", type=Path)
    create.add_argument("manifest", type=Path)
    create.add_argument("--label", default="")
    create.add_argument("--source-pdf-name", default="")
    create.add_argument("--page-spec", default="")
    create.add_argument("--baseline-checkpoint", default="")

    verify = subparsers.add_parser("verify", help="verify an output directory")
    verify.add_argument("output_dir", type=Path)
    verify.add_argument("manifest", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Execute a manifest create or verify command.

    执行清单创建或校验命令，并返回适合shell使用的退出码。
    """

    args = build_parser().parse_args(argv)
    if args.command == "create":
        manifest = write_manifest(
            args.output_dir,
            args.manifest,
            label=args.label,
            source_pdf_name=args.source_pdf_name,
            page_spec=args.page_spec,
            baseline_checkpoint=args.baseline_checkpoint,
        )
        print(f"wrote {manifest['file_count']} file records: {args.manifest}")
        return 0

    errors = verify_manifest(args.output_dir, args.manifest)
    if errors:
        for error in errors:
            print(error)
        return 1
    print(f"manifest verified: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
