# 第三方组件与资源声明

本文件说明 NativePDF Structurer 仓库中不适用项目 Apache License 2.0，或仍需
遵守额外许可证条件的组件与资源。

## GapTree Sort Algorithm

- 上游项目：GapTree Sort Algorithm
- 作者：hiroi-sora
- 许可证：MIT License
- 仓库内文件：`nativePDF_structurer/_vendor/gap_tree.py`

本项目的阅读顺序处理以该开源实现为基础，并针对 PDF 文字、表格、图片和矢量
区域的统一排序进行了扩展。完整 MIT 许可证见
`nativePDF_structurer/_vendor/GapTree_LICENSE.txt`。Apache License 2.0 不替代
该第三方代码原有的 MIT 许可证和署名要求。

## PyMuPDF / MuPDF

PyMuPDF 是通过 Python 包管理器安装的外部运行时依赖，并未由本仓库重新授权。
PyMuPDF 官方采用 GNU Affero General Public License v3（AGPL-3.0）或 Artifex
商业许可双授权。安装、使用或分发 NativePDF Structurer 与 PyMuPDF 时，使用者
仍需根据自己的使用和分发方式遵守适用的 PyMuPDF / MuPDF 许可条款；本项目的
Apache License 2.0 不会解除或改变这些义务。

其他通过 `pyproject.toml` 或 `requirements.txt` 安装的依赖仍分别受其上游
许可证约束。

## 文档示例图片

`docs/assets/examples/` 中的图片来自厂商技术文档的转换结果，仅用于展示解析器的
输出形态。相关原始内容、图形和商标的权利归各自权利人所有，这些示例图片不在
本项目 Apache License 2.0 的再授权范围内。

---

# Third-Party Components and Assets

This file identifies components and assets in the NativePDF Structurer
repository that are not covered by the project's Apache License 2.0, or that
remain subject to additional license terms.

## GapTree Sort Algorithm

- Upstream project: GapTree Sort Algorithm
- Author: hiroi-sora
- License: MIT License
- Repository file: `nativePDF_structurer/_vendor/gap_tree.py`

The reading-order processing in this project is based on that open-source
implementation and extends it for shared ordering of PDF text, tables, images,
and vector regions. The complete MIT license is retained in
`nativePDF_structurer/_vendor/GapTree_LICENSE.txt`. Apache License 2.0 does not
replace the original MIT license or attribution requirements for this
third-party code.

## PyMuPDF / MuPDF

PyMuPDF is an external runtime dependency installed through the Python package
manager and is not relicensed by this repository. PyMuPDF is officially dual
licensed under the GNU Affero General Public License v3 (AGPL-3.0) or an
Artifex commercial license. Anyone installing, using, or distributing
NativePDF Structurer with PyMuPDF must still comply with the applicable
PyMuPDF / MuPDF terms for their use and distribution model. The project's
Apache License 2.0 does not remove or alter those obligations.

Other dependencies installed through `pyproject.toml` or `requirements.txt`
remain subject to their respective upstream licenses.

## Documentation example images

Images in `docs/assets/examples/` are converted excerpts from manufacturer
technical documents and are included only to demonstrate parser output. The
original content, graphics, and trademarks remain the property of their
respective rights holders. These example images are not relicensed under the
project's Apache License 2.0.
