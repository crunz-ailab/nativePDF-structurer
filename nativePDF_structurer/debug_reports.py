"""生成解析诊断使用的人工可读报告。

模块将已收集的页面、Claim、几何、标题、
图片和阅读顺序记录整理为精简Markdown；报告生成是只读操作，不改变解析、
归属、渲染或输出选择。

Generate human-readable reports for parser diagnostics.

Formats already collected page, Claim, geometry, heading, image, and reading-
order records as concise Markdown reports. Report generation is read-only and
does not alter parsing, ownership, rendering, or output selection.
"""



from __future__ import annotations

import io



from collections import Counter, defaultdict

from pathlib import Path

import fitz
from PIL import Image, ImageDraw



#@GPT 报告模块只消费已完成的记录字典，不参与PDF检测、Claim仲裁或页面状态修改。

#@GPT 调试报告使用稳定文件名和Markdown字段，且不会影响最简模式的文档输出。

def write_logical_horizontal_rule_report(
    path: Path,
    records: list[dict],
) -> None:
    """输出原始drawing片段到逻辑横线及最终归属的Markdown清单。"""
    status_labels = {
        "kept_for_table_and_visual_structure": "保留",
        "dropped_from_logical_view_below_width_threshold": (
            "低于15%，仅逻辑视图丢弃"
        ),
        "excluded_from_table_detector_by_preclassified_rule_band": (
            "说明带预分类已摘除"
        ),
        "raw_source_drawings_available_to_table_detector": (
            "原始片段进入表格检测"
        ),
        "logical_rule_not_created_below_width_threshold": (
            "未进入视觉逻辑线"
        ),
        "excluded_from_visual_by_preclassified_rule_band": (
            "说明带已认领"
        ),
        "excluded_from_visual_by_table_skeleton": (
            "表格骨架已认领"
        ),
        "available_to_visual_relation_and_framework_detection": (
            "可用于视觉关系/框架"
        ),
    }
    merged_count = len(records)
    kept_count = sum(
        record.get("passed_logical_width_filter", False)
        for record in records
    )
    lines = [
        "# 同 Y 水平片段合并审计",
        "",
        (
            f"共记录 **{merged_count}** 条由两个以上原始 drawing "
            f"拼接的逻辑横线，其中 **{kept_count}** 条在合并后通过 "
            "15% 页面宽度门槛。"
        ),
        "",
        "> 原始 drawing ID 始终保留；逻辑横线只是一份结构视图，不生成替代原图元的新 ID。",
        "",
    ]

    records_by_page: dict[int | None, list[dict]] = defaultdict(list)
    for record in records:
        records_by_page[record.get("page_number")].append(record)

    for page_number in sorted(
        records_by_page,
        key=lambda value: value if value is not None else -1,
    ):
        lines.extend(
            [
                f"## Page {page_number}",
                "",
                "| 逻辑横线 | 原始 drawing ID | 原始片段 x范围@y | 合并后范围 | 页面宽度占比 | 逻辑视图 | 表格来源状态 | 视觉逻辑状态 | 阶段D最终 owner |",
                "|---|---|---|---|---:|---|---|---|---|",
            ]
        )
        for record in records_by_page[page_number]:
            source_ranges = "<br>".join(
                (
                    f"{rect[0]:.2f}–{rect[2]:.2f}"
                    f"@{(rect[1] + rect[3]) / 2:.2f}"
                )
                for rect in record.get("source_rects", [])
            )
            merged_bbox = record["merged_bbox"]
            merged_range = (
                f"{merged_bbox[0]:.2f}–{merged_bbox[2]:.2f}"
                f"@{record['merged_y']:.2f}"
            )
            owner_text = "<br>".join(
                record.get("final_owner_claim_ids", [])
            ) or "—"
            lines.append(
                "| "
                f"`{record['logical_rule_id']}` | "
                f"{', '.join(record['source_drawing_ids'])} | "
                f"{source_ranges} | "
                f"{merged_range} | "
                f"{record['merged_width_page_ratio']:.2%} | "
                f"{status_labels.get(record['logical_view_status'], record['logical_view_status'])} | "
                f"{status_labels.get(record['table_source_status'], record['table_source_status'])} | "
                f"{status_labels.get(record['visual_logical_status'], record['visual_logical_status'])} | "
                f"{owner_text} |"
            )
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

#@GPT GapTree调试图渲染只消费页面与排序元数据；DPI由核心显式传入，避免报告模块复制策略常量。
def render_gaptree_debug_image(
    page: fitz.Page,
    page_number: int,
    gaptree_metadata: dict,
    output_dir: Path,
    *,
    debug_dpi: int,
) -> Path:
    """渲染页面并叠加GapTree节点边框、编号和节点间连接线。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    pixmap = page.get_pixmap(
        dpi=debug_dpi,
        alpha=False,
    )
    image = Image.open(io.BytesIO(pixmap.tobytes("png"))).convert(
        "RGB"
    )
    draw = ImageDraw.Draw(image)
    scale = debug_dpi / 72.0
    colors = [
        "#d62728",
        "#1f77b4",
        "#ff7f0e",
        "#9467bd",
        "#2ca02c",
        "#8c564b",
    ]
    centers: list[tuple[float, float]] = []

    for node in gaptree_metadata.get("nodes", []):
        x0, y0, x1, y1 = node["bbox"]
        scaled_bbox = (
            x0 * scale,
            y0 * scale,
            x1 * scale,
            y1 * scale,
        )
        node_index = node["node_index"]
        color = colors[(node_index - 1) % len(colors)]
        draw.rectangle(scaled_bbox, outline=color, width=3)
        label = str(node_index)
        label_bbox = draw.textbbox(
            (scaled_bbox[0] + 2, scaled_bbox[1] + 2),
            label,
        )
        draw.rectangle(label_bbox, fill="white", outline=color)
        draw.text(
            (scaled_bbox[0] + 2, scaled_bbox[1] + 2),
            label,
            fill=color,
        )
        centers.append(
            (
                (scaled_bbox[0] + scaled_bbox[2]) / 2,
                (scaled_bbox[1] + scaled_bbox[3]) / 2,
            )
        )

    #@GPT 不绘制节点中心连线：跨栏跳转会形成长对角线遮挡正文，节点编号已经表达前序顺序。

    output_path = output_dir / (
        f"page_{page_number:03d}_gaptree_nodes.png"
    )
    image.save(output_path)
    return output_path

#@GPT 阅读顺序人工报告兼容GapTree和内部XY-Cut元数据，不参与节点生成或顺序裁决。
def write_gaptree_report(
    path: Path,
    pages: list[dict],
) -> None:
    """写出比原始JSON更易读的空间布局报告。"""
    lines = [
        "# 阅读顺序空间布局报告",
        "",
        "节点编号表示最终后端节点顺序。",
    ]

    for page_record in pages:
        lines.extend(
            [
                "",
                f"## Page {page_record['page_number']}",
                "",
                (
                    f"调试图："
                    f"[`{Path(page_record['gaptree_debug_image']).name}`]"
                    f"(gaptree_debug/{Path(page_record['gaptree_debug_image']).name})"
                ),
            ]
        )

        for direction in page_record["text_reflow"][
            "direction_groups"
        ]:
            metadata = direction["gaptree"]
            fragment_merge = metadata.get("fragment_merge", {})
            #@GPT 报告新增列首回挂摘要，让人工核查时能直接看到是否启用、学习到几列以及移动了多少元素。
            column_correction = metadata.get(
                "column_anchor_correction",
                {},
            )
            lines.extend(
                [
                    "",
                    f"### 方向 {direction['angle']}°",
                    "",
                    f"后端：`{metadata.get('backend', 'gaptree')}`",
                    "",
                    (
                        f"行 {metadata['row_count']}；切线 "
                        f"{metadata['cut_count']}；布局节点 "
                        f"{metadata['node_count']}。"
                    ),
                    (
                        "同行碎片合并："
                        f"{'启用' if fragment_merge.get('enabled') else '未启用'}；"
                        f"合并 {fragment_merge.get('merged_fragment_count', 0)} 个碎片；"
                        f"原因 `{fragment_merge.get('reason', 'not_applicable')}`。"
                    ),
                    (
                        "列锚点校正："
                        f"{'启用' if column_correction.get('enabled') else '未启用'}；"
                        "稳定列 "
                        f"{column_correction.get('stable_column_count', 0)}；"
                        "移动 "
                        f"{column_correction.get('moved_element_count', 0)} 个元素；"
                        "原因 "
                        f"`{column_correction.get('reason', 'not_applicable')}`。"
                    ),
                    "",
                    "| 顺序 | bbox | 元素类型 | 样例 |",
                    "|---:|---|---|---|",
                ]
            )

            for node in metadata["nodes"]:
                samples = " / ".join(node["samples"]).replace(
                    "|", "\\|"
                )
                lines.append(
                    f"| {node['node_index']} | `{node['bbox']}` | "
                    f"`{node['kinds']}` | {samples} |"
                )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

#@GPT Claim冲突报告只格式化阶段C-F的现有记录，不重新执行仲裁。
def write_claim_conflict_report(
    path: Path,
    page_resolutions: list[dict],
) -> None:
    """输出阶段C声明及阶段D旁路仲裁的人工可读报告。"""
    lines = [
        "# 检测器声明冲突报告",
        "",
        #@GPT 报告分别展示阶段D候选存活、阶段E结构边界和阶段F文字归属结果。
        # "> 阶段 D 已按唯一归属过滤最终候选；阶段 E 的 render_bbox 重建尚未启用。",
        "> 阶段 D 已按唯一归属过滤最终候选；阶段 E/F 已按获胜图元和文字归属重建视觉区域。",
        "",
    ]

    for page_record in page_resolutions:
        page_number = page_record["page_number"]
        resolution = page_record["resolution_preview"]
        claims = page_record["claims"]
        lines.extend(
            [
                f"## Page {page_number}",
                "",
                (
                    f"候选 **{len(claims)}** 个；声明图元 "
                    f"**{resolution['claimed_element_count']}** 个；冲突图元 "
                    f"**{resolution['conflict_element_count']}** 个；无效ID引用 "
                    f"**{resolution['invalid_element_reference_count']}** 个。"
                ),
                "",
                "### 候选存活预览",
                "",
                #@GPT 新增存活原因列，低置信候选因结构被拆散退出时无需再到动作JSON中反查。
                "| 候选 | 检测器 | 置信度 | 声明图元 | 特征图元 | 赢得特征 | 存活 | 原因 |",
                "|---|---|---:|---:|---:|---:|---|---|",
            ]
        )
        claims_by_id = {
            claim["claim_id"]: claim for claim in claims
        }
        for candidate in resolution["candidates"]:
            claim = claims_by_id[candidate["claim_id"]]
            lines.append(
                "| "
                f"`{candidate['claim_id']}` | "
                f"{candidate['detector']} | "
                f"{claim['confidence']:.2f} | "
                f"{candidate['claimed_count']} | "
                f"{candidate['feature_count']} | "
                f"{candidate['won_feature_count']} | "
                f"{'是' if candidate['survives_preview'] else '否'} | "
                f"{candidate.get('survival_reason', '—')} |"
            )

        #@GPT 人工报告并列展示检测器提议bbox和获胜feature IDs并集bbox，便于核查边界变化。
        lines.extend(
            [
                "",
                "### 阶段 E 边界预览",
                "",
                "| 候选 | 检测器提议 bbox | 获胜特征 bbox | 面积比 |",
                "|---|---|---|---:|",
            ]
        )
        for candidate in resolution["candidates"]:
            proposed_bbox_text = ", ".join(
                f"{value:.1f}"
                for value in candidate.get("proposed_bbox", [])
            )
            owned_bbox = candidate.get("owned_feature_bbox")
            owned_bbox_text = (
                ", ".join(f"{value:.1f}" for value in owned_bbox)
                if owned_bbox is not None
                else "—"
            )
            area_ratio = candidate.get(
                "owned_to_proposed_area_ratio"
            )
            area_ratio_text = (
                f"{area_ratio:.3f}"
                if area_ratio is not None
                else "—"
            )
            lines.append(
                f"| `{candidate['claim_id']}` | "
                f"[{proposed_bbox_text}] | "
                f"[{owned_bbox_text}] | "
                f"{area_ratio_text} |"
            )

        stage_f_preview = resolution.get(
            "stage_f_text_preview", {}
        )
        lines.extend(
            [
                "",
                "### 阶段 F 文字归属预览",
                "",
                (
                    f"已分配文字 **{stage_f_preview.get('assigned_text_count', 0)}** 行；"
                    f"保留正文 **{stage_f_preview.get('unassigned_text_count', 0)}** 行；"
                    f"多候选竞争 **{stage_f_preview.get('conflict_text_count', 0)}** 行。"
                ),
                "",
                "| 候选 | 检测器 | 分配文字行 |",
                "|---|---|---:|",
            ]
        )
        for candidate in stage_f_preview.get("candidates", []):
            lines.append(
                f"| `{candidate['claim_id']}` | "
                f"{candidate['detector']} | "
                f"{candidate['assigned_text_count']} |"
            )

        lines.extend(["", "### 冲突图元", ""])
        if not resolution["conflicts"]:
            lines.extend(["本页没有多检测器声明同一图元。", ""])
            continue

        lines.extend(
            [
                "| 图元ID | 类型 | bbox | 拟归属 | 裁决原因 | 竞争候选 |",
                "|---|---|---|---|---|---|",
            ]
        )
        for conflict in resolution["conflicts"]:
            contenders = "<br>".join(
                (
                    f"{item['detector']} / `{item['claim_id']}` "
                    f"(P={item['priority']}, C={item['confidence']:.2f})"
                )
                for item in conflict["contenders"]
            )
            bbox_text = ", ".join(
                f"{value:.1f}" for value in conflict["bbox"]
            )
            lines.append(
                "| "
                f"`{conflict['element_id']}` | "
                f"{conflict['kind']} | "
                f"[{bbox_text}] | "
                f"`{conflict['winner_claim_id']}` | "
                f"{conflict.get('resolution_reason', 'unknown')}"
                f"{'（同检测器）' if conflict.get('same_detector_conflict') else ''} | "
                f"{contenders} |"
            )
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

#@GPT 页眉页尾模板序列化与报告放在同一输出模块；运行期开关由核心显式传入。
def serialize_header_footer_templates(
    templates: dict[str, dict],
) -> dict[str, dict]:
    """将页眉/页尾内部模板转换为精简可序列化结构。"""
    output: dict[str, dict] = {}

    for edge in ["header", "footer"]:
        edge_data = templates.get(edge, {})
        output[edge] = {
            "enabled": bool(edge_data.get("enabled")),
            "sample_page_numbers": edge_data.get(
                "sample_page_numbers", []
            ),
            "minimum_page_support": edge_data.get(
                "minimum_page_support", 0
            ),
            "minimum_matched_objects": edge_data.get(
                "minimum_matched_objects", 0
            ),
            "cooccurring_pages": edge_data.get(
                "cooccurring_pages", []
            ),
            "templates": [
                {
                    "kind": template["kind"],
                    "bbox": [
                        round(value, 3)
                        for value in template["bbox"]
                    ],
                    "height": round(template["height"], 3),
                    #@GPT 精简模板补充宽度和复合成员数；页尾水平线与Logo均可核对是否按完整对象学习。
                    "width": round(
                        template.get("width", 0.0), 3
                    ),
                    "x0": round(template["x0"], 3),
                    "edge_offset": round(
                        template["edge_offset"], 3
                    ),
                    "outer_offset": (
                        round(template["outer_offset"], 3)
                        if template.get("outer_offset")
                        is not None
                        else None
                    ),
                    "outer_side": template.get("outer_side"),
                    "composite_member_count": int(
                        template.get(
                            "composite_member_count", 1
                        )
                    ),
                    "content_signature_prefix": template[
                        "content_signature"
                    ][:12],
                }
                for template in edge_data.get("templates", [])
            ],
        }

    return output

def write_header_footer_report(
    path: Path,
    templates: dict[str, dict],
    pages: list[dict],
    *,
    detect_footer: bool,
    keep_header_text_in_markdown: bool,
) -> None:
    """输出页眉/页尾硬匹配模板及逐页结果。"""
    serialized_templates = serialize_header_footer_templates(
        templates
    )
    lines = [
        "# 页眉与页尾硬匹配报告",
        "",
        #@GPT 页尾镜像模板不再要求绝对X一致；报告说明页眉与页尾各自采用的硬位置语义，避免调试时误认为奇偶页走了页码特例。
        "文字内容不参与模板匹配；页眉使用绝对X位置，页尾复合对象使用距最近外侧页边的位置，二者均硬匹配尺寸、距页边高度和内容摘要。",
        "",
        (
            "页尾检测：停用。"
            if not detect_footer
            else "页尾检测：启用。"
        ),
        (
            "页眉文字回写：启用，字段格式为 `页眉:[页眉文本]`。"
            if keep_header_text_in_markdown
            else "页眉文字回写：停用。"
        ),
    ]

    for edge, label in [("header", "页眉"), ("footer", "页尾")]:
        edge_data = serialized_templates[edge]
        lines.extend(
            [
                "",
                f"## {label}模板",
                "",
                f"启用：{'是' if edge_data['enabled'] else '否'}",
                "",
                f"样本页：`{edge_data['sample_page_numbers']}`",
                "",
                f"共同匹配页：`{edge_data['cooccurring_pages']}`",
                "",
                "| 类型 | bbox | 宽×高 | X0 | 外侧距离 | 距上下页边 | 成员数 | 内容摘要 |",
                "|---|---|---:|---:|---:|---:|---:|---|",
            ]
        )

        for template in edge_data["templates"]:
            lines.append(
                f"| {template['kind']} "
                f"| `{template['bbox']}` "
                f"| {template['width']}×{template['height']} "
                f"| {template['x0']} "
                f"| {template['outer_offset']} "
                f"| {template['edge_offset']} "
                f"| {template['composite_member_count']} "
                f"| `{template['content_signature_prefix']}` |"
            )

    lines.extend(
        [
            "",
            "## 逐页结果",
            "",
            "| 页面 | 页眉匹配对象 | 页眉排除带 | 页尾匹配对象 | 页尾排除带 |",
            "|---:|---:|---|---:|---|",
        ]
    )

    for page in pages:
        lines.append(
            f"| {page['page_number']} "
            f"| {page['header']['matched_object_count']} "
            f"| `{page['header']['band']}` "
            f"| {page['footer']['matched_object_count']} "
            f"| `{page['footer']['band']}` |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

#@GPT 标题报告显式接收三个开关，避免调试输出模块读取核心全局配置。
def write_heading_detection_report(
    path: Path,
    typography_profile: dict,
    records: list[dict],
    *,
    enable_heading_detection: bool,
    enable_pdf_outline_heading_assignment: bool,
    enable_font_heading_inference: bool,
) -> None:
    """写出文档级字号画像和最终标题清单。"""
    lines = [
        "# Markdown 标题检测报告",
        "",
        #@GPT 标题总开关与字体推断开关分别展示，避免目录匹配启用而字体推断关闭时产生误导。
        # f"启用：{'是' if typography_profile.get('enabled') else '否'}",
        f"标题总开关：{'开启' if enable_heading_detection else '关闭'}",
        "",
        (
            "目录严格文本匹配："
            f"{'开启' if enable_pdf_outline_heading_assignment else '关闭'}"
        ),
        "",
        (
            "字体/字号推断："
            f"{'开启' if enable_font_heading_inference else '关闭（默认）'}"
        ),
        "",
        (
            "正文基准字号："
            f"`{typography_profile.get('body_font_size')}` pt"
        ),
        "",
        (
            "大于正文的字号簇："
            f"`{typography_profile.get('heading_size_clusters', [])}`"
        ),
        "",
        f"最终输出标题：**{len(records)}** 行。",
        "",
        #@GPT 报告增加本页正文参考字号和书签命中项，便于直接判断“为什么这行比正文大”以及层级是否来自PDF显式目录。
        #@GPT 新增Markdown输出位置，区分由页面H2承载的首目录标题和仍在正文bbox输出的同页其他目录标题。
        "| 页面 | MD等级 | 输出位置 | 字号 | 本页正文上限 | 粗体字符占比 | 原因 | PDF书签 | 文字匹配 | 文本 |",
        "|---:|---:|---|---:|---:|---:|---|---|---|---|",
    ]

    for record in records:
        escaped_text = (
            record["text"]
            .replace("|", "\\|")
            .replace("\n", " ")
        )
        lines.append(
            f"| {record['page_number']} "
            f"| H{record['level']} "
            f"| `{record.get('markdown_emission', 'body_heading')}` "
            f"| {record['effective_font_size']} "
            f"| {record.get('body_reference_font_size')} "
            f"| {record['bold_character_ratio']:.3f} "
            f"| `{record['reason']}` "
            f"| {record.get('outline_title') or ''} "
            f"| {record.get('outline_text_matches')} "
            f"| {escaped_text} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def write_visual_region_report(
    path: Path,
    records: list[dict],
) -> None:
    """将视觉区域记录输出为精简 Markdown。"""
    lines = [
        "# 矢量视觉区域处理清单",
        "",
        f"共生成 **{len(records)}** 个粗粒度视觉区域。",
        "",
        #@GPT 报告增加端点附着列，便于人工核对上下/斜向引线是否接管了文字。
        "| 页面 | 图片文件 | 矢量对象 | 初始连通域 | 纳入文字行 | 端点文字 | 纳入小图片 | 图片尺寸 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]

    for record in records:
        filename = Path(record["output_path"]).name
        lines.append(
            f"| {record['page_number']} "
            f"| [`{filename}`](visual_regions/{filename}) "
            f"| {record['drawing_count']} "
            f"| {record['component_count']} "
            f"| {record['included_text_line_count']} "
            f"| {record.get('endpoint_promoted_text_line_count', 0)} "
            f"| {record['included_image_count']} "
            f"| {record['output_width']} × {record['output_height']} px |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def write_formula_region_report(
    path: Path,
    records: list[dict],
) -> None:
    """输出默认公式渲染结果的人工可读 Markdown 清单。"""
    lines = [
        "# 数学公式渲染清单",
        "",
        f"共生成 **{len(records)}** 个显示公式页面渲染图。",
        "",
        "| 页面 | 图片文件 | 分栏 | 编号锚点 | 归属文字行 | 图片尺寸 |",
        "|---:|---|---|---|---:|---:|",
    ]

    for record in records:
        filename = Path(record["output_path"]).name
        labels = ", ".join(record["equation_labels"]) or "行内编号"
        number_anchor = (
            labels if record["has_number_anchor"] else "无"
        )
        lines.append(
            f"| {record['page_number']} "
            f"| [`{filename}`](formula_images/{filename}) "
            f"| {record['column']} "
            f"| {number_anchor} "
            f"| {record['owned_text_line_count']} "
            f"| {record['output_width']} × {record['output_height']} px |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def write_source_image_report(
    path: Path,
    records: list[dict],
) -> None:
    """输出只包含实际生成图片及其核心判定信息的人工可读清单。"""
    mode_labels = {
        "direct_copy": "直接复制原图",
        "rendered_composite": "复合区域截图",
    }
    reason_labels = {
        "multiple_images_overlap": "多张图片相互叠加",
        "text_overlay": "文字叠加",
        "vector_drawing_overlap": "矢量图形叠加",
        "annotation_overlap": "PDF 批注叠加",
        "form_widget_overlap": "表单控件叠加",
        "soft_mask": "包含软蒙版",
        "complex_transform": "存在旋转或倾斜变换",
        "missing_original_bytes": "缺少可直接复制的原始字节",
    }
    saved_records = [
        record
        for record in records
        if record.get("output_path")
    ]
    skipped_counts = Counter(
        record["mode"]
        for record in records
        if not record.get("output_path")
    )
    mode_counts = Counter(record["mode"] for record in saved_records)

    lines = [
        "# 原 PDF 图片处理清单",
        "",
        (
            f"实际生成 **{len(saved_records)}** 张图片："
            f"直接复制 **{mode_counts['direct_copy']}** 张，"
            f"复合区域截图 **{mode_counts['rendered_composite']}** 张。"
        ),
        "",
        (
            "下表只保留与 `source_images` 目录中实际文件有关的信息；"
            "不展示内部对象索引、绝对路径、坐标和逐图片计时。"
        ),
    ]

    if skipped_counts:
        lines.extend(
            [
                "",
                #@GPT 精简图片报告同时统计表格和视觉区域覆盖数量，说明图片为何未重复输出。
                (
                    "未逐条列出："
                    f"小装饰图片 {skipped_counts['filtered_small_image']} 个，"
                    f"已由表格截图覆盖 {skipped_counts['covered_by_table']} 个，"
                    "已由粗视觉区域覆盖 "
                    f"{skipped_counts['covered_by_visual_region']} 个。"
                ),
            ]
        )

    records_by_page: dict[int, list[dict]] = defaultdict(list)

    for record in saved_records:
        records_by_page[record["page_number"]].append(record)

    for page_number in sorted(records_by_page):
        lines.extend(
            [
                "",
                f"## Page {page_number}",
                "",
                "| 图片文件 | 处理方式 | 判定依据 | 图片尺寸 |",
                "|---|---|---|---:|",
            ]
        )

        for record in records_by_page[page_number]:
            filename = Path(record["output_path"]).name
            reasons = record.get("reason") or []
            reason_text = "、".join(
                reason_labels.get(reason, reason)
                for reason in reasons
            )

            if not reason_text:
                reason_text = "没有检测到叠加元素"

            dimensions = (
                f"{record['output_width']} × "
                f"{record['output_height']} px"
            )
            lines.append(
                f"| [`{filename}`](source_images/{filename}) "
                f"| {mode_labels.get(record['mode'], record['mode'])} "
                f"| {reason_text} | {dimensions} |"
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def write_figure_caption_report(
    path: Path,
    records: list[dict],
) -> None:
    """写出最终进入Markdown的图题及其唯一视觉区域归属。"""
    lines = [
        "# Markdown 图题检测报告",
        "",
        (
            "图题检测在表格、视觉区域和原图归属完成后执行；"
            "只处理未被其他区域认领、带通用图号结构且与一个视觉区域垂直相邻的单行文字。"
        ),
        "",
        f"最终输出图题：**{len(records)}** 行。",
        "",
        "| 页面 | 图题 | 目标Claim | 区域号 | 方位 | 垂直间隔(pt) | X重叠(pt) | 原因 |",
        "|---:|---|---|---:|---|---:|---:|---|",
    ]

    for record in records:
        escaped_text = (
            str(record.get("text", ""))
            .replace("|", "\\|")
            .replace("\n", " ")
        )
        lines.append(
            f"| {record['page_number']} "
            f"| {escaped_text} "
            f"| `{record.get('target_claim_id')}` "
            f"| {record.get('target_region_number')} "
            f"| `{record.get('side')}` "
            f"| {record.get('gap_points')} "
            f"| {record.get('x_overlap_points')} "
            f"| `{record.get('reason')}` |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
