# 工程架构

**简体中文** | [English](architecture_EN.md)

NativePDF Structurer 是一个面向数字原生 PDF 的零 OCR、零模型依赖结构化器。
它的输出不只是 Markdown 转换结果，而是：

```text
Markdown + 引用资源 + 结构化 IR + provenance
```

解析器直接使用 PDF 中已有的文字、字体、图片、drawing 和坐标层，不要求
OCR、GPU 或外部语言模型。

## 工程结构

下面的目录树展示发布仓库中的主要内容。为避免与后文的模块职责清单重复，
同类模块合并在一行表示，未逐个展开全部测试文件：

```text
nativePDF-structurer/
├── README.md / README_EN.md          # 中文主说明与英文说明
├── pyproject.toml                    # 包元数据、依赖范围和 CLI 入口
├── LICENSE / THIRD_PARTY_NOTICES.md  # 项目许可证与第三方授权边界
├── environment.yml / requirements.txt
│                                      # Conda 环境与验证依赖版本
├── nativePDF_structurer/             # Python 主包
│   ├── __init__.py / __main__.py / cli.py
│   │                                  # 公共 API、模块入口和命令行接口
│   ├── config.py / run_state.py / page_context.py
│   │                                  # 文档配置、文档状态和单页状态
│   ├── pipeline.py                    # 文档与页面阶段总编排
│   ├── inventory.py / claims.py
│   ├── output_ownership.py / text_ownership.py
│   ├── cross_detector_ownership.py    # 图元清单、Claim 和跨检测器归属
│   ├── table_detection.py / table_structure.py
│   ├── table_skeleton.py / table_output.py
│   │                                  # 表格检测、结构、骨架和输出
│   ├── visual_assets.py / visual_processing.py
│   │                                  # 原始图片、矢量区域及附着关系
│   ├── layout_geometry.py / geometry.py
│   ├── rule_geometry.py / rule_bands.py / rule_preclassification.py
│   │                                  # 版面几何与规则线预分类
│   ├── reading_order_backend.py       # 页面原子的空间阅读顺序
│   ├── header_footer.py / heading_detection.py
│   ├── outline_headings.py / figure_captions.py / formula_regions.py
│   │                                  # 页边模板、标题、图题和公式区域
│   ├── page_reflow.py / markdown_formatter.py
│   ├── artifacts.py / emitter.py / ir.py / result.py
│   │                                  # Markdown、资源、IR 和最终结果
│   ├── page_contract.py / debug_reports.py / regression_manifest.py
│   │                                  # 调试契约、审计报告和输出回归
│   └── _vendor/
│       ├── gap_tree.py                # 内置阅读顺序基础实现
│       └── GapTree_LICENSE.txt        # 对应的 MIT 许可证
├── docs/
│   ├── architecture.md / architecture_EN.md
│   │                                  # 中英文工程架构文档
│   └── assets/examples/               # README 使用的精简转换示例
├── tests/                              # 通用单元测试和契约测试
└── .github/workflows/tests.yml         # GitHub Actions 测试流程
```

这棵树用于回答“代码大致放在哪里”。具体算法边界、状态所有权和各模块职责，
以后文说明为准。

## 运行路径

```text
CLI / RunConfig
    -> pipeline
    -> 文档运行状态
    -> pipeline / 每页一个 PageRunContext
    -> 检测器与所有权阶段回调
    -> 页面提交
    -> ParseResult / ArtifactEmitter
```

运行路径完全位于 `nativePDF_structurer` 包内。开发 checkpoint 和本地回归
资产不会进入发布源码，运行时也不会导入历史单文件解析器。

## 状态归属

| 对象 | 负责保存 | 不应保存 |
|---|---|---|
| `ParseRunContext` | 已验证的文档选项和从 1 开始的选定页码 | 可变页面对象 |
| `DocumentRunState` | 已打开文档、输出布局、共享缓存、计时指标、目录和字体统计状态 | 其他页面的候选所有权 |
| `PageRunContext` | 当前页的原始/工作对象及全部页面中间结果 | 文档级结果列表 |
| `ParseResult` | 已完成页面、Markdown 章节、IR 输入和 provenance 集合 | 检测器临时状态 |

`PageRunContext.raw_drawings` 保存未经修改的页面图元清单；`drawings` 是确认
过滤页眉页尾矢量后的工作视图。两者同时存在，可以防止后续检测器把已过滤对象
误认为源 PDF 中不存在。

## 页面阶段

调试模式下，每页都会记录以下阶段历史：

```text
initialization
  -> header_footer
  -> geometry
  -> rule_bands
  -> table_detection
  -> table_candidates
  -> table_finalization
  -> visual_protection
  -> visual_detection
  -> output_proposal
  -> source_images
  -> claim_resolution
  -> boundary_ownership
  -> semantics
  -> reading_order
  -> finalization
```

阶段名称只在 `nativePDF_structurer.page_contract.PageStage` 中定义一次。若某个
阶段被跳过、重复执行或顺序发生变化，调试契约会报告不一致。

## 模块边界

模块按可维护的算法主题划分，避免重新拆成大量只包含少数函数的小文件：

- `claims.py`：规范化 Claim 清单并定义检测器优先级策略。
- `geometry.py`：提供矩形、中心点、并集和 drawing item 等共享几何原语。
- `inventory.py`：生成稳定页面图元 ID，维护逻辑容器，按需展开 drawing item，
  并解决 Claim 冲突。
- `output_ownership.py`：在冲突解决后判断输出是否存活，并抑制父子视觉区域的
  重复输出。
- `text_ownership.py`：在结构边界确定后执行阶段 F 的原生文字归属。
- `reading_order_backend.py`：提供可替换的阅读顺序接口、虚拟正向坐标变换、
  同行窄列附着、跨列首行修复和水平文字碎片合并。它还执行第二阶段横向
  XY-Cut，用于将视觉对象与其下方短文本带分开，同时保持复合视觉组；最终再
  应用“分隔符位于标题之前”的显式顺序规则。视觉区域、原始图片和预分类规则
  带内部保留的原生文字也由同一排序器处理。
- `table_detection.py`：表格检测的粗粒度入口，包含小掩码预聚簇、原生检测、
  几何补检、fallback、局部框架检测、候选去重、页面级编排、结构候选构建和
  Table Claim 准备。阈值与解析回调通过显式依赖对象传入，不直接读取 pipeline
  的全局变量。
- `table_structure.py`：负责表格候选策略、Claim 构建、纯几何单元格提取、
  原生/fallback 表格兼容对象及合并、规则线位置分组、原生单元格边界检查、
  父容器 Claim 和表格局部证据审计。复杂内部 drawing 不会因此被扣分：完整
  截图表格可以认领 bbox 内部内容，而父容器表格可以把子视觉内容留给后续
  所有权阶段。
- `table_skeleton.py`：独立保存性能敏感的大簇轴向骨架算法。拓扑和预算选择
  逻辑集中在这里，便于性能分析和修改表格几何，而不影响其他模块。
- `table_output.py`：负责 CSV、纯文本 Markdown 网格、方向推断、矩阵规范化、
  CSV 旋转、直角 PNG 无损旋转以及 PNG/Markdown 表格原子输出。纯文本表格
  直接输出 Markdown 的路径保持为可选项；复杂视觉表格继续使用截图路径。
- `cross_detector_ownership.py`：处理跨检测器的 overlap 关系、视觉文本块键、
  source cluster 连接证据、表格/视觉复合输出组、阶段 D 复合结果确认和独立
  表格输出恢复。它协调表格、视觉区域和表格输出记录，但不转移原始图元所有权，
  也不直接解决 inventory 冲突。
- `pipeline.py`：可编辑的总编排边界，包含页面阶段顺序、具体阶段适配器、选定
  页面循环以及最终 summary、IR 和 `ArtifactEmitter` 处理。将这些紧密相关的
  工作保留在同一模块，可以让后续算法修改保持局部化，避免形成分散的回调图。
- `visual_assets.py`：负责原始图片输入、图片 block 规范化、尺寸/变换过滤、
  crop padding、内容摘要、矢量候选过滤、掩码坐标、封闭视觉容器检测、OpenCV
  初始区域构建以及矢量与原始图片候选合并。它还包含文字附着几何判据、一跳
  连接件提升、图内原生文字 Markdown 格式化，以及基于 `render_bbox`、
  `sort_bbox` 构建视觉输出原子的逻辑。
- `header_footer.py`：提取局部页眉页尾候选，组合相邻页尾 drawing，执行确定性
  硬匹配，并从文档采样页分别学习页眉和页尾模板。逐页匹配只过滤确认命中的
  drawing，同时可以选择将原生页眉文字保留到 Markdown。
- `formula_regions.py`：依据原生字体和符号证据提取公式行，判断双栏布局，垂直
  聚合候选并附着公式编号；随后完成候选接受、页面渲染、Markdown 原子构建和
  精确公式文字归属。
- `figure_captions.py`：根据标签结构、垂直邻近、竞争正文排除和“一图一题”规则，
  将外部图题行分配给最终视觉输出。
- `page_contract.py`：执行仅调试模式启用的阶段、Claim、bbox 和资源一致性检查。
- `page_commit.py` / `page_finalize.py`：把完成的页面状态提交到文档级结果。
- `artifacts.py` / `emitter.py` / `ir.py`：管理输出布局、调试资源、结构化中间
  表示和 provenance。
- `markdown_formatter.py`：把已经排序的页面原子转换为 Markdown，负责标题、
  图题、非文字原子、文字缓冲和语义分隔符渲染，不读取 PDF 几何或检测器状态。
- `page_reflow.py`：过滤已经分类的文字行，按方向分组，调用注入的阅读顺序和
  几何判据，并追加稳定的方向性 Markdown 章节。它不负责 PDF 解析或检测决策。
- `rule_geometry.py`：把原始 drawing path item 转换为轻量轴向线段，合并精确
  共线片段，构建页面级横向/纵向逻辑规则线，并记录审计映射。
- `rule_bands.py` / `rule_preclassification.py`：检测视觉框架带，并在表格检测前
  预分流成对水平规则带。模块只处理注入的几何证据和阈值，不执行 OCR 或重复
  读取页面。
- `debug_reports.py`：输出视觉区域、原始图片、公式、Claim 冲突、逻辑规则线、
  标题、图题和页眉页尾模板的人类可读调试报告；最简输出模式不会调用它。
- `layout_geometry.py`：负责粗粒度原生文字提取、PDF 目录页名、规则线拓扑、
  文字分隔符、矩形 overlap、空间索引和预聚簇对象关系。
- `visual_processing.py`：处理原始图片、矢量端点/标记规则、视觉区域编排、
  预分类视觉带渲染，以及阶段 E/F 的视觉区域与原图边界重建。

页面和文档阶段顺序、阶段适配器以及最终输出都由 `pipeline.py` 编排。
`pipeline.py` 同时构建文档级共享资源和各检测器使用的显式依赖对象。运行路径
不再依赖历史单文件实现。

视觉资源边界覆盖图片输入、矢量候选过滤、掩码原语、封闭容器检测、初始连通域
构建、结构候选合并和几何文字附着。跨检测器输出关系由
`cross_detector_ownership.py` 处理；截图、视觉输出记录、视觉 Claim 和粗粒度
视觉区域编排属于视觉处理边界。

## Claim 与所有权模型

每个检测器 Claim 区分三组 ID：

- `input_element_ids`：检测器检查过的对象；
- `claimed_element_ids`：候选希望拥有的对象；
- `feature_element_ids`：一旦全部丢失，候选就应失效的特征对象。

`inventory.py` 依据检测器优先级、置信度、特征数量和稳定 Claim ID 解决结构
冲突。文字归属延迟到结构边界确定后执行，避免同一文字行同时被表格、视觉区域
和图片检测器认领。

输出原子带有两个空间框：

- `render_bbox`：完整截图或 crop 范围；
- `sort_bbox`：用于阅读顺序的锚点，通常是顶部的一条窄带。

后续修改不能把这两个 bbox 合并为同一个字段。

## 调试契约

调试模式会检查以下内容，但不会改变正常输出：

- 页面阶段顺序完全一致；
- Claim ID 存在且唯一，feature/claimed 集合关系有效；
- inventory owner 引用有效；
- `bbox`、`render_bbox` 和 `sort_bbox` 非反向、位于页面范围内；
- Markdown 输出文件、provenance 记录和资源路径存在；
- 活跃输出路径不重复。

最简模式只写入正常 Markdown 和引用资源。调试 JSON、报告、阶段历史及契约
警告均为可选输出。

## 安全扩展流程

修改检测器或所有权规则时：

1. 除非有意修改 schema，否则保持稳定图元 ID 和 Claim 字段不变。
2. 修改检测器或模块策略，不要把算法判断放进页面提交或 emitter 层。
3. 保持 `render_bbox` 与 `sort_bbox` 的不同语义。
4. 为新增几何或所有权情况添加专项单元测试。
5. 在覆盖相关几何结构的聚焦页面上运行调试契约。
6. 先运行聚焦 manifest，再运行配置范围的 full-range manifests。
7. 在把结果交给视觉检查前保存 checkpoint。

本地回归可以使用通用 manifest 命令：

```bash
python -m nativePDF_structurer.regression_manifest verify \
  /path/to/parser-output /path/to/manifest.json
```

项目专用 PDF、页码选择、manifest 和视觉问题记录应放在私有本地回归目录，而
不是发布源码树。字节级 manifest 可以验证输出契约，但相同 hash 不能证明页面
布局在语义上更合理，因此仍需要视觉检查。

## 阅读顺序后端

包对外只暴露一个窄的阅读顺序后端契约。它接收虚拟正向坐标中的页面原子，
返回排序后的原子和版面元数据：

```text
页面原子 -> 已排序原子 + 版面元数据
```

任何替代实现都必须保持 Markdown、非文字资源 hash、图元顺序和页面级
provenance，才能进入正式发布包。
