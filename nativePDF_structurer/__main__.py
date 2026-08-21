"""支持通过 ``python -m nativePDF_structurer`` 启动公开命令行接口。

本模块只
转交给 ``nativePDF_structurer.cli``，不包含独立的解析或配置逻辑。

Run the public command-line interface with ``python -m``.

Delegates directly to :mod:`nativePDF_structurer.cli` and contains no parsing
or configuration logic of its own.
"""

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
