# Audio Agent 框架

基于 LangGraph 的音频理解框架，支持迭代调用工具。

## 概览

本框架为构建音频理解智能体提供清晰架构，能够：

- 使用前端 LALM（大音频语言模型）处理音频
- 使用 LLM 规划器对证据进行推理并决定下一步动作
- 迭代调用工具以收集更多证据
- 汇总并融合来自多源的证据
- 产出带支撑证据的最终答案

## 架构

```
START
  -> frontend_evidence_node（LALM 处理音频，生成初始描述）
  -> initial_plan_node（仅依据问题进行规划）
  -> planner_decision_node（LLM 决定动作；最后一步强制 ANSWER）
  -> [根据决策条件分支]
     - ANSWER -> evidence_summarization_node（对所有证据的中性摘要）
       -> final_answer_node（前端模型基于音频与上下文生成答案）
       -> format_check_node（强制校验）
         * 格式通过 -> answer_node -> END
         * 格式失败 -> planner_decision_node（将批评作为证据继续循环）
     - CALL_TOOL -> tool_executor_node（自动注入 audio_path）
       -> evidence_fusion_node -> planner_decision_node（循环）
     - CLARIFY_INTENT -> intent_clarification_node -> planner_decision_node（循环）
     - FAIL -> failure_node -> END
```

**关键行为：**

- **初始规划**：规划器仅根据问题生成高层次方案。
- **证据摘要**：在最终答案之前，由文本 LLM 将所有证据、规划轨迹与工具历史压缩为单一中性叙事，避免前端模型被冗长的原始工具输出淹没。
- **前端最终答案**：由具备音频能力的前端模型直接根据原始音频与摘要上下文生成最终答案，而非由文本规划器直接写出答案。
- **格式检查**：在定稿前进行强制格式校验；若格式不符，将批评加入证据并继续规划。
- **工具契约**：底层信号/元数据类工具不得覆盖前端的语义判断。

## 项目结构

```
audio_agent/
├── __init__.py
├── main.py                 # 主入口与 AudioAgent 类
├── core/                   # 核心类型与工具
│   ├── state.py           # AgentState 定义
│   ├── schemas.py         # Pydantic 模型
│   ├── errors.py          # 自定义异常
│   ├── constants.py       # 枚举与常量
│   └── logging.py         # 日志工具
├── frontend/              # 音频前端实现
│   ├── base.py            # BaseFrontend 抽象基类
│   ├── model_frontend.py  # BaseModelFrontend 模板
│   ├── dummy_frontend.py  # 占位实现
│   ├── qwen2_audio_frontend.py      # Qwen2-Audio 适配（本地）
│   ├── qwen3_omni_frontend.py       # Qwen3-Omni 适配（本地）
│   └── openai_compatible_frontend.py # OpenAI 兼容 API 前端
├── planner/               # 规划器实现
│   ├── base.py            # BasePlanner 抽象基类
│   ├── model_planner.py   # BaseModelPlanner 模板
│   ├── dummy_planner.py   # 占位实现
│   └── qwen25_planner.py  # Qwen2.5 规划器适配
├── tools/                 # 工具系统
│   ├── base.py            # BaseTool 抽象基类
│   ├── registry.py        # ToolRegistry（内置 + MCP 工具）
│   ├── executor.py        # ToolExecutor
│   ├── dummy_tools.py     # 占位工具
│   ├── mcp/               # MCP（模型上下文协议）基础设施
│   │   ├── client.py      # MCP 客户端
│   │   ├── server_manager.py  # MCP 服务生命周期
│   │   ├── tool_adapter.py    # MCP 到 BaseTool 的适配
│   │   └── schemas.py     # MCP 数据模型
│   └── catalog/           # MCP 工具目录
│       ├── loader.py      # 自动发现并注册
│       ├── _template/     # 新工具模板
│       ├── asr_qwen3/     # Qwen3-ASR-1.7B 语音识别
│       ├── diarizen/      # 说话人分离
│       └── omni_captioner/ # Qwen3-Omni 描述生成
├── fusion/                # 证据融合
│   ├── base.py           # BaseEvidenceFuser 抽象基类
│   └── default_fuser.py  # 默认实现
├── graph/                 # LangGraph 工作流
│   ├── builder.py        # 图构建
│   ├── nodes.py          # 节点函数
│   └── routing.py        # 路由逻辑
├── log/                   # 运行日志
│   ├── __init__.py       # 导出：RunLogger、log_run
│   ├── logger.py         # Markdown 日志的 RunLogger
│   └── formatter.py      # Markdown 格式化工具
├── prompts/               # Markdown 提示词文件
│   ├── frontend_system.md           # 前端系统提示
│   ├── frontend_user.md             # 前端用户指令
│   ├── frontend_final_answer_system.md  # 前端最终答案系统提示
│   ├── frontend_final_answer_user.md    # 前端最终答案用户指令
│   ├── plan_system.md               # 规划器规划系统提示
│   ├── plan_user.md                 # 规划器规划用户指令
│   ├── decide_system.md             # 规划器决策系统提示
│   ├── decide_user.md               # 规划器决策用户指令
│   ├── decide_rules.md              # 规划器决策规则
│   ├── clarify_system.md            # 规划器澄清系统提示
│   ├── clarify_user.md              # 规划器澄清用户指令
│   ├── format_check_system.md       # 格式检查系统提示
│   ├── format_check_user.md         # 格式检查用户指令
│   ├── evidence_summary_system.md   # 证据摘要系统提示
│   ├── evidence_summary_user.md     # 证据摘要用户指令
│   └── task_skills.yaml             # 初始规划用的任务技能参考
├── config/                # 配置
│   └── settings.py       # AgentConfig
├── utils/                 # 工具函数
│   ├── validation.py     # 校验辅助
│   ├── model_io.py       # 模型 I/O 辅助
│   ├── prompt_io.py      # 提示词加载
│   └── model_downloader.py  # 模型下载工具
├── examples/              # 示例脚本
│   ├── demo_run.py                # 基础演示（本地模型）
│   ├── demo_run_auto_tools.py     # 自动发现 MCP 工具的演示
│   ├── demo_run_real_asr.py       # 真实 ASR 工具演示
│   ├── demo_run_api_planner.py    # API 规划器 + 本地前端
│   └── demo_run_api_full.py       # API 前端 + API 规划器（无需 GPU）
└── tests/                 # 测试
    ├── test_state.py
    ├── test_registry.py
    ├── test_graph_smoke.py
    └── ...
```

## 安装

```bash
# 创建并激活 conda 环境
conda create -n audio_agent python=3.11
conda activate audio_agent

# 以可编辑模式安装本包
pip install -e .

# 或直接安装依赖
pip install -r requirements.txt
```

## 运行演示

演示使用真实模型（Qwen2-Audio 前端、Qwen2.5 规划器）并自动发现 MCP 工具：

```bash
# 先配置 MCP 工具（需要 uv）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 分别安装各工具
cd audio_agent/tools/catalog/asr_qwen3 && ./setup.sh && cd -
cd audio_agent/tools/catalog/diarizen && ./setup.sh && cd -
cd audio_agent/tools/catalog/omni_captioner && ./setup.sh && cd -

# 或使用辅助脚本一次性安装全部工具
./verify_all_tools.sh --setup

# 下载模型
audio-agent-download-models --models qwen2-audio qwen2.5 qwen3-asr

# 单段音频 + 自动发现工具
python -m audio_agent.examples.demo_run_auto_tools \
  --audio /path/to/audio.wav \
  --question "What is being said in this audio?"

# 多段音频（例如说话人比对）
python -m audio_agent.examples.demo_run_auto_tools \
  --audio /path/to/audio1.wav --audio /path/to/audio2.wav \
  --question "Is the speaker in the second audio any of the speakers in the first audio?"
```

### 校验工具环境

确认所有 MCP 工具已正确配置：

```bash
# 测试全部工具
./verify_all_tools.sh

# 安装并测试全部工具
./verify_all_tools.sh --setup
```

指定 ASR 配置的演示见 `demo_run_real_asr.py`。

### 基于 API 的演示（无需本地 GPU）

若无本地 GPU 或希望使用 API 模型：

```bash
# 先同样安装 MCP 工具
./verify_all_tools.sh --setup

# API 规划器 + 本地前端（单段音频）
export DASHSCOPE_API_KEY="sk-xxx"
python -m audio_agent.examples.demo_run_api_planner \
  --audio /path/to/audio.wav \
  --question "What is being said?"

# API 前端 + API 规划器（完全走 API，无本地模型）
python -m audio_agent.examples.demo_run_api_full \
  --audio /path/to/audio.wav \
  --question "What is being said?" \
  --frontend-model "qwen3-omni-flash" \
  --planner-model "qwen3.5-plus"

# 多段音频 API 示例（说话人验证）
python -m audio_agent.examples.demo_run_api_full \
  --audio /path/to/first_audio.wav --audio /path/to/second_audio.wav \
  --question "Is the speaker in the second audio the same as the first?" \
  --frontend-model "qwen3-omni-flash" \
  --planner-model "qwen3.5-plus"
```

`demo_run_api_full.py` 适合：

- 没有本地 GPU 的用户
- 快速原型与测试
- 推理由外部服务承载的部署场景

## 预下载模型

默认使用本地模型路径，避免每次登录重复下载。模型默认存放在 `/lihaoyu/workspace/AUDIO_AGENT/models/`。

**一次性下载全部模型：**

```bash
pip install -e ".[download]"
audio-agent-download-models --all
```

**只下载部分模型：**

```bash
audio-agent-download-models --models qwen2-audio qwen2.5
```

**列出可用模型及状态：**

```bash
audio-agent-download-models --list
```

**可用模型：**

- `qwen2-audio` - Qwen/Qwen2-Audio-7B-Instruct（前端，约 15GB）
- `qwen3-omni` - Qwen/Qwen3-Omni-30B-A3B-Instruct（前端，约 60GB）
- `qwen2.5` - Qwen/Qwen2.5-7B-Instruct（规划器，约 15GB）
- `qwen3-asr` - Qwen/Qwen3-ASR-1.7B（ASR 工具，约 4GB）
- `qwen3-aligner` - Qwen/Qwen3-ForcedAligner-0.6B（对齐工具，约 1.5GB）
- `diarizen` - BUT-FIT/diarizen-wavlm-large-s80-md（说话人分离，约 1GB）
- `omni-captioner` - Qwen/Qwen3-Omni-30B-A3B-Captioner（描述器，约 60GB）

**使用 HuggingFace Hub 路径（备选）：**

若直接使用 Hub 路径，模型会下载到缓存目录：

```bash
# 单段音频
python -m audio_agent.examples.demo_run \
  --audio /path/to/audio.wav \
  --question "What is being said?" \
  --frontend-model-path Qwen/Qwen2-Audio-7B-Instruct \
  --planner-model-path Qwen/Qwen2.5-7B-Instruct

# 多段音频对比
python -m audio_agent.examples.demo_run \
  --audio /path/to/audio1.wav --audio /path/to/audio2.wav \
  --question "Compare the speakers in these two audio files" \
  --frontend-model-path Qwen/Qwen2-Audio-7B-Instruct \
  --planner-model-path Qwen/Qwen2.5-7B-Instruct
```

## 运行测试

```bash
pytest audio_agent/tests/ -v
```

## 多音频支持

单次运行可处理多个音频文件，适用于：

- **说话人验证**：判断两段音频是否包含同一说话人
- **音频对比**：在多文件间对比内容、质量或特征
- **多源分析**：联合分析来自不同源的音频

当提供多个音频时：

- 每段音频分配 ID：`audio_0`、`audio_1`、`audio_2` 等
- 前端会处理全部音频并为每段生成描述
- 工具可按 ID 引用特定音频
- 规划器可推理音频之间的关系

### 示例：说话人验证

```python
from audio_agent.main import create_api_full_agent

agent = create_api_full_agent(
    frontend_model="qwen3-omni-flash",
    planner_model="qwen3.5-plus",
)

# 以列表传入多段音频
result = agent.run(
    question="Is the speaker in the second audio any of the speakers in the first audio?",
    audio_paths=["/path/to/first_audio.wav", "/path/to/second_audio.wav"]
)

if agent.is_successful(result):
    print(result["final_answer"].answer)
```

## 设计原则

### 快速失败（Fail-Fast）

- 处理前校验全部输入
- 缺少必填字段立即抛出明确异常
- 非法工具名、畸形输出即刻捕获
- 无静默回退或对结构化结果返回 `None`

### 显式契约

- 各组件有明确的输入/输出模式
- 使用 Pydantic 校验
- 全面使用类型注解

### 可扩展性

- 主要组件均为抽象基类
- 易于将占位实现替换为真实实现
- 通过工厂函数实现清晰依赖注入

### 职责分离

- 前端：初始音频理解
- 规划器：决策
- 工具：外部能力
- 融合：证据累积
- 图：工作流编排

## 扩展框架

### 添加新工具

```python
from audio_agent.tools.base import BaseTool
from audio_agent.core.schemas import ToolSpec, ToolCallRequest, ToolResult

class MyAudioTool(BaseTool):
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="my_audio_tool",
            description="Does something with audio",
            input_schema={"type": "object", "properties": {...}},
            output_schema={"type": "object", "properties": {...}},
        )

    def invoke(self, request: ToolCallRequest) -> ToolResult:
        # 在此实现逻辑
        return ToolResult(
            tool_name=self.spec.name,
            success=True,
            output={"result": "..."},
        )

# 注册工具
registry.register(MyAudioTool())
```

### 添加真实前端

**本地模型前端：**

```python
from audio_agent.frontend.base import BaseFrontend
from audio_agent.core.schemas import FrontendOutput

class RealLALMFrontend(BaseFrontend):
    def __init__(self, model_path: str):
        self.model = load_model(model_path)

    @property
    def name(self) -> str:
        return "real_lalm"

    def run(self, question: str, audio_paths: list[str]) -> FrontendOutput:
        self.validate_inputs(question, audio_paths)
        # 在此实现逻辑
        return FrontendOutput(
            question_guided_caption="...",
        )
```

**基于 API 的前端：**

使用内置 `OpenAICompatibleFrontend`：

```python
from audio_agent.frontend.openai_compatible_frontend import OpenAICompatibleFrontend

frontend = OpenAICompatibleFrontend(
    model="qwen3-omni-flash",  # 或任意支持音频的 API 模型
    api_key="sk-xxx",  # 或设置环境变量 DASHSCOPE_API_KEY
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
```

API 前端将音频以 base64 编码发送并接收文本描述。

### 添加真实规划器

```python
from audio_agent.planner.base import BasePlanner
from audio_agent.core.schemas import PlannerDecision

class OpenAIPlanner(BasePlanner):
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)

    @property
    def name(self) -> str:
        return "openai_planner"

    def plan(self, question: str) -> InitialPlan:
        # 仅根据问题生成初始计划
        return InitialPlan(approach="Analyze audio content...")

    def decide(self, state, available_tools) -> PlannerDecision:
        self.validate_state(state)
        # 在此实现逻辑
        return PlannerDecision(...)

    def summarize_evidence(self, state) -> str:
        # 将累积证据摘要为中性叙事
        return "Summary of evidence..."

    def check_format(self, proposed_answer, expected_format, question) -> FormatCheckResult:
        # 仅校验格式合规性
        return FormatCheckResult(passed=True)
```

### 使用 MCP 工具

MCP 工具在独立进程中运行，支持自动发现：

```python
import asyncio
from audio_agent.main import AudioAgent
from audio_agent.tools.catalog import register_all_mcp_tools
from audio_agent.tools.mcp import MCPServerManager
from audio_agent.frontend.qwen2_audio_frontend import Qwen2AudioFrontend
from audio_agent.planner.qwen25_planner import Qwen25Planner
from audio_agent.tools.registry import ToolRegistry
from audio_agent.fusion.default_fuser import DefaultEvidenceFuser

async def run_with_tools():
    frontend = Qwen2AudioFrontend()
    planner = Qwen25Planner()
    registry = ToolRegistry()
    fuser = DefaultEvidenceFuser()

    server_manager = MCPServerManager()
    await register_all_mcp_tools(registry, server_manager, verbose=True)

    agent = AudioAgent(frontend, planner, registry, fuser)
    result = await agent.arun(
        question="What is being said?",
        audio_paths=["/path/to/audio.wav"]
    )

    await server_manager.shutdown_all()
    return result

asyncio.run(run_with_tools())
```

### 自定义提示词

提示词已全部外置到 `audio_agent/prompts/` 下的 Markdown 文件中，可直接编辑以调整规划器与前端行为。

**提示文件一览：**

| 文件 | 用途 | 变量 |
|------|------|------|
| `frontend_system.md` | 前端系统提示 | 无 |
| `frontend_user.md` | 前端用户指令 | `{question}`、`{audio_path_or_uri}` |
| `frontend_final_answer_system.md` | 前端最终答案系统提示 | 无 |
| `frontend_final_answer_user.md` | 前端最终答案用户指令 | `{question}`、`{expected_output_format}`、`{initial_plan_text}`、`{frontend_direct_text}`、`{evidence_and_history_text}`、`{audio_summary}`、`{format_critique_section}` |
| `plan_system.md` | 规划器初始规划系统提示 | 无 |
| `plan_user.md` | 规划器初始规划用户指令 | `{question}` |
| `decide_system.md` | 规划器决策系统提示 | 无 |
| `decide_user.md` | 规划器决策用户指令 | `{question}`、`{frontend_caption}`、`{initial_plan}`、`{evidence_log}`、`{tool_call_history}`、`{available_tools}`、`{step_count}`、`{max_steps}` |
| `decide_rules.md` | 规划器决策规则 | 无 |
| `clarify_system.md` | 规划器澄清系统提示 | 无 |
| `clarify_user.md` | 规划器澄清用户指令 | `{question}`、`{clarified_intent}`、`{expected_format}`、`{evidence_text}` |
| `format_check_system.md` | 格式检查系统提示 | 无 |
| `format_check_user.md` | 格式检查用户指令 | `{question}`、`{expected_format}`、`{proposed_answer}` |
| `evidence_summary_system.md` | 证据摘要系统提示 | 无 |
| `evidence_summary_user.md` | 证据摘要用户指令 | `{question}`、`{frontend_caption}`、`{evidence_text}`、`{planner_trace_text}`、`{tool_history_text}`、`{clarified_intent}`、`{expected_output_format}` |
| `task_skills.yaml` | 初始规划的任务技能参考 | 渲染为 Markdown「菜谱」 |

**示例：自定义前端系统提示**

编辑 `audio_agent/prompts/frontend_system.md`：

```markdown
You are an expert audio analyst. Focus on identifying speakers, emotions,
and background sounds relevant to the question.
Return ONLY the caption as plain text.
```

**示例：增加决策规则**

编辑 `audio_agent/prompts/decide_rules.md` 以添加自定义决策逻辑：

```markdown
1. If you have enough evidence to answer the question, use action='answer'.
2. If you need transcription, use action='call_tool' with an ASR tool.
3. If you need speaker information, use action='call_tool' with a diarization tool.
...
```

**在代码中加载提示词：**

```python
from audio_agent.utils.prompt_io import load_prompt

system_prompt = load_prompt("frontend_system")
user_prompt = load_prompt("plan_user").format(question="What is being said?")
```

### 添加 MCP 工具

新工具接入推荐使用 **Harness-First Agent 工作流**：

```bash
cat tool_preparation/README.md
```

手动开发请参阅 [Tool 准备指南](./tool_preparation/README.md)。简要步骤：

```bash
# 1. 复制模板
cp -r audio_agent/tools/catalog/_template audio_agent/tools/catalog/my_tool

# 2. 编辑 pyproject.toml、server.py、config.yaml

# 3. 编写 setup.sh 与 test_env.sh（模板见 tool_preparation/playbooks/env_uv.md）

# 4. 创建环境
cd audio_agent/tools/catalog/my_tool && ./setup.sh

# 5. 校验
./test_env.sh

# 6. 注册并测试
./verify_all_tools.sh
```

## 许可证

MIT
