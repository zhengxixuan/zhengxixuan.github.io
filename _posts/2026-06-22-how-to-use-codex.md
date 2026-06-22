---
title: 像高手一样使用 Codex
description: 如何把 Codex 放进一个可控、可验证、可复用、可累积迭代的 AI 工程工作流里？
date: 2026-06-22 20:30:00 +0800
categories: [AI, Reflection]
tags: [practice]
---

不要把 Codex 只当成“代码自动补全工具”，而要把它当成一个有终端权限的初中级软件工程队友。

你的任务不是亲自写每一行代码，而是：

**定义任务、限制范围、检查 diff、要求验证、决定是否合并。**

真正的高手不是让 Codex “随便写代码”，而是为 Codex 建立一套稳定的自主工作系统。

**如何把 Codex 放进一个可控、可验证、可复用、可累积迭代的 AI 工程工作流里？**

我们先来看看一些高级技巧：

## Codex 自主运行的 4 个高级技巧

### 1. 手机监督：让 Codex 在你离开电脑后继续推进

核心玩法是：你在电脑上运行 Codex App，然后用手机上的 ChatGPT App 远程启动、查看、批准、调整 Codex 任务。

Codex mobile 已经在 iOS 和 Android 的 ChatGPT App 中预览推出，可以在手机上 start、steer、approve、check in，让 Codex 继续在 laptop、Mac mini 或 remote computer 上工作。

这件事听起来像小功能，但实际改变很大。

过去，开发者必须坐在电脑前，盯着命令行和编辑器。现在，Codex 可以在机器上继续工作，而你只需要在关键节点做判断。

一个真实场景可能是：

你出门前让 Codex 修一个失败测试。
路上用手机查看进度。
Codex 请求安装新依赖。
你拒绝，并补充一句：

```text
不要新增依赖。请先检查项目里是否已有类似工具函数，只做最小修改。
```

过一会儿，你再次打开手机。Codex 已经完成修改，跑了相关测试，并汇报了剩余风险。

这不是“用手机写代码”。

这是用手机调度一个正在运行的工程代理。

---

### 2. Worktrees：让多个 Codex 任务并行运行

近期 Codex 相关视频和官方材料都在强调 app/worktree/parallel agents。OpenAI 对 Codex App 的介绍中明确提到，它是一个管理多个 agents 的 command center，支持多个线程并行、内置 worktrees，避免多个 agent 修改同一 repo 时互相冲突。

Codex app 很重要的能力，是可以通过 worktrees 隔离不同任务。

这意味着你可以让多个 Codex 线程同时探索不同方向，而不必把所有改动混在同一个工作区里。

比如：

- 一个线程修 bug。
- 一个线程补测试。
- 一个线程尝试重构方案。
- 一个线程只做代码审查。

它们可以各自在独立 worktree 中运行，互不污染。你最后只需要比较它们的 diff、测试结果和风险说明，再决定采用哪个方案。

这对工程实践很关键。

并行不是为了让 Codex 同时乱改，而是为了让探索过程变得可隔离、可比较、可回退。

过去，一个开发者通常一次只能认真推进一个方向。现在，你可以让 Codex 同时生成几个候选路径，然后由人来判断和选择。

这体现了 Codex app 的另一个核心优势：它不只是帮你写代码，而是在扩展你的工程探索能力。

---

### 3. AGENTS.md 和 Skills：让 Codex 的行为可以被沉淀

如果每次使用 Codex，你都要重复提醒它：

不要新增依赖。不要大规模重构。修改行为必须补测试。最终要说明风险。遵循项目已有风格。

那说明这些规则不应该只写在 prompt 里，而应该沉淀下来。

**AGENTS.md 可以看成项目里的“AI 协作说明书”。**它告诉 Codex，这个项目如何安装、如何测试、有哪些代码风格、哪些事情不能做、完成任务前必须汇报什么。

一个简单版本可以这样写：

```markdown
# AGENTS.md

## 项目原则
- 优先最小修改，不做无关重构。
- 不新增依赖，除非明确解释原因。
- 不修改 public API，除非任务明确要求。
- 行为变更必须配测试。

## 常用命令
- 安装：pnpm install
- 测试：pnpm test
- 类型检查：pnpm typecheck
- Lint：pnpm lint

## Codex 完成任务前必须说明
- 修改了哪些文件；
- 运行了哪些测试；
- 哪些风险没有验证；
- 是否有需要人工确认的地方。

```

**Skill 是一组 instructions、resources 和 optional scripts，让 Codex 能稳定执行某个任务流。**

Skills 更像固定任务流程。比如 bugfix、PR review、release note、frontend polish，都可以做成 skill。

这类用法很适合你未来做自己的 AI coding workflow。比如你可以做几个固定 skills：

```
- pr-review：按你的工程审查标准检查 PR；
- regression-test-writer：根据 bug 自动补 regression test；
- frontend-polish：检查 UI 层级、间距、响应式和 accessibility；
- release-note-writer：根据 git diff 生成 release notes；
- architecture-reader：阅读陌生 repo 并生成架构地图；
- ai-course-demo-builder：把教学 demo 自动整理成可运行项目。
```

一个很实用的 skill 描述可以这样写：

```
---
name: minimal-bugfix
description: 用于修复生产 bug，要求最小 diff、可复现、可验证。
---

工作流程：
1. 先复现 bug，不要急着改代码。
2. 找到最小根因路径。
3. 只修改必要文件。
4. 添加 regression test。
5. 运行相关测试。
6. 最后输出 root cause、modified files、tests run、remaining risk。
```

AGENTS.md 解决“这个项目的一般规则”。
Skills 解决“这类任务应该怎么做”。

这让 Codex 的使用不再是一次性的。你今天纠正过的错误，明天可以变成规则；你今天验证过的流程，明天可以变成 skill。

一个自主运行系统最重要的特征，不只是会执行，而是能被约束、被训练、被持续改进。

---

### 4. Tests、Browser 和 Hooks：让自主运行保持可验证、可控

自主运行不等于放任运行。

Codex 越能自动推进任务，越需要验证和护栏。

**测试是第一层验证。**你应该要求 Codex 修改后运行相关测试、lint 或 typecheck，并清楚说明哪些已经验证，哪些没有验证。

对于前端任务，测试还不够。页面是否真的正常，交互是否顺畅，布局是否错位，需要打开真实页面看。**Codex 可以通过 browser 查看本地页面，观察结果，再进行修复。**

例如：

```text
使用 browser 打开 http://localhost:3000/dashboard。
请不要先改代码。

先完成：
1. 观察当前页面布局问题；
2. 截图并描述视觉 bug；
3. 找到最可能相关的组件；
4. 提出最小修复计划。

确认后再修改。
```

**Hooks 则是工程护栏。你可以把一些规则自动化：**

实际玩法包括：

```
- 每次 Codex 修改文件后自动跑 lint；
- 每次执行 shell command 前检查是否危险；
- 如果修改 package.json，自动要求人工批准；
- 如果改动超过 N 个文件，自动停止并要求总结；
- 每次任务结束自动生成 diff summary；
- 每次通过测试后自动生成 commit message。
```

Codex hooks、remote SSH、mobile control 组合后，Codex 已经像一个“工程操作层”  。hooks 是 Codex 的 extensibility framework，可以把你自己的脚本注入 agentic loop。

这三项能力很重要，因为它回答了一个核心问题：

如果 Codex 可以自主运行，如何保证它不会越界？

答案不是盲目信任，而是把它放进可验证、可中断、可审查的流程里。

Codex 的成熟用法，不是让它随便跑，而是让它在边界内自主推进。

*接下来的部分，推荐给新手。如果你从未使用过 Codex，可以优先了解它的基础用法。*

## 用好 Codex 的 9 个要点


### 1. 选择合适的 Codex 使用场景

如果你正在 IDE 里处理某个具体文件，适合用 **Codex IDE extension**。它能理解你当前打开的文件、选中的代码和局部上下文。

如果任务需要运行命令、执行测试、检查项目结构、修改多个文件，适合用 **Codex CLI**。它更像一个可以在本地项目中工作的 coding agent。

如果你想把一个较大的任务交给 Codex，让它在云端环境里完成，并最终形成一个 PR，可以使用 **Codex web / cloud**。

如果你同时管理多个开发任务，可以使用 **Codex app**，把不同任务拆成不同 thread 或 worktree 来处理。

简单说：

```text
小范围代码修改：IDE extension
本地开发与调试：CLI
较大任务与 PR：Codex cloud
多任务并行管理：Codex app
```

---

### 2. 用四个字段写 Prompt

好的 Codex prompt 通常包含四个部分：

**目标、上下文、约束、完成标准。**

模板如下：

```text
目标：
实现 [具体改动]。

上下文：
相关文件：[文件路径]
当前行为：[现在发生了什么]
期望行为：[应该发生什么]
错误 / 日志 / 复现步骤：[粘贴具体信息]

约束：
- 遵循现有架构。
- 不要随意添加新依赖。
- 保持 public API 兼容。
- 控制 diff 范围。
- 添加或更新测试。

完成标准：
- [指定测试命令] 通过。
- [指定行为] 被验证。
- 解释最终改动和潜在取舍。
```

差的 prompt：

```text
Fix the auth bug.
```

高手 prompt：

```text
目标：
修复 session 过期后登录页面反复重定向的问题。

上下文：
相关文件：
- src/auth/session.ts
- src/routes/login.tsx
- src/middleware/auth.ts

复现步骤：
1. 登录。
2. 删除 session cookie。
3. 访问 /dashboard。

当前结果：
页面在 /login 和 /dashboard 之间循环重定向。

期望结果：
只重定向一次到 /login?expired=1。

约束：
- 不要改变 OAuth provider flow。
- 保持现有 middleware 结构。
- 添加一个 regression test。

完成标准：
- npm test -- auth 通过。
- npm run lint 通过。
- 先总结 root cause，再展示 diff。
```

---

### 3. 把 `AGENTS.md` 当成项目里的“AI 协作宪法”

不要每次都重复告诉 Codex 项目规则。把稳定规则写进 `AGENTS.md`。

它可以包括：

```md
# AGENTS.md

## 项目概览
这是一个 [Next.js / Python / CLI / etc.] 项目。请保持改动小而清晰，并遵循现有模式。

## 常用命令
- 安装依赖：pnpm install
- 启动开发：pnpm dev
- 测试：pnpm test
- 类型检查：pnpm typecheck
- Lint：pnpm lint

## 工程规则
- 优先使用已有工具函数，不要轻易新增抽象。
- 不要随意添加生产依赖；如果必须添加，需要解释原因。
- 除非任务明确要求 breaking change，否则保持 public API 兼容。
- 行为变更必须添加或更新测试。
- 结束前先运行最相关的窄范围测试，再视情况运行更广泛检查。

## Review 要求
最终回复前，请说明：
- 修改了哪些文件；
- 运行了哪些测试；
- 还有哪些风险或未验证区域。
```

高手做法是：
**Codex 同一个错误犯两次，就把规则写进 `AGENTS.md`。**

例如：

```md
- 不要重写整个组件，只做最小必要修改。
- 不要修改数据库 schema，除非任务明确要求。
- 不要添加全局状态管理库。
- 所有日期处理必须使用项目已有的 date utils。
```

这会让 Codex 越用越贴合你的项目。

---

### 4. 复杂任务先让 Codex 制定计划，不要直接改代码

对于复杂、模糊、风险较大的任务，第一步不要让 Codex 写代码，而是让它先读项目、分析问题、提出计划。

可以这样说：

```text
先不要编辑代码。请先检查相关文件并提出计划。

计划中需要包含：
1. 可能的 root cause；
2. 需要修改的文件；
3. 测试策略；
4. 风险；
5. 最小实现路径。

等我确认后再开始编码。
```

确认计划后，再说：

```text
按照最小安全实现路径继续。请保持 diff 聚焦。
运行相关测试。如果测试失败，请先尝试定位并修复一次，然后再交还给我。
```

这可以避免 Codex 一上来就大改项目，甚至重写半个 repo。

---

### 5. 强制验证，而不是只让它生成代码

Codex 最有价值的地方不是“生成代码”，而是它可以：

```text
修改代码
运行测试
观察错误
继续修复
解释 diff
```

所以 prompt 里要明确要求验证：

```text
实现后请运行：
1. 与本次改动最相关的窄范围测试；
2. 相关 typecheck / lint；
3. 对自己的 diff 做一次 self-review。

最终回复中请包含：
- 改了什么；
- 运行了哪些测试；
- 还有什么风险；
- 哪些地方是你有意没有改的。
```

这一步非常关键。
不经过测试的 Codex 输出，只能算“草稿代码”。

---

### 6. 把任务拆成适合 agent 的粒度

不要轻易说：

```text
Build the whole app.
```

除非你只是想快速做一个原型。

生产级任务应该拆小：

```text
任务 1：阅读当前架构并提出计划。
任务 2：只实现 data model。
任务 3：只实现 API route。
任务 4：只实现 UI。
任务 5：添加测试。
任务 6：审查完整 diff，找回归风险。
```

更好的拆法是按边界拆：

```text
- 数据层
- API 层
- UI 层
- 测试
- 文档
- Review
```

不要让多个 Codex 线程同时修改同一批文件。
并行可以用，但要用于互不冲突的任务。

---

### 7. 并行 agent 适合做调查，不适合同时乱改

并行 Codex 很适合做 read-only investigation。

例如：

```text
请启动多个只读 agent，分别调查：

1. 安全风险；
2. 性能瓶颈；
3. flaky tests；
4. API 兼容性；
5. 前端可访问性。

不要修改文件。等所有 agent 返回后，汇总发现。
```

这种用法很强，因为它让多个 agent 从不同角度看同一份代码。

但不要这样用：

```text
Agent A 改 auth
Agent B 改 middleware
Agent C 改 routing
Agent D 改 tests
```

如果这些任务都碰同一批文件，很容易互相覆盖、制造冲突。

---

### 8. 把重复工作沉淀成 Skills

如果你发现自己经常让 Codex 做同一类事情，就应该把它做成 Skill。

适合沉淀为 Skill 的工作包括：

```text
- PR review checklist
- 前端设计检查
- bug 复现流程
- 数据分析报告流程
- release note 生成
- migration checklist
- 测试生成流程
```

一个简单的 PR Review Skill 可以这样写：

```md
---
name: pr-review
description: 用于审查 pull request，发现 bug、回归、缺失测试和可维护性风险。
---

Review 优先级：
1. 正确性 bug；
2. 安全 / 隐私问题；
3. 缺失测试；
4. 行为回归；
5. 可维护性问题。

输出要求：
- 先列发现的问题，按严重程度排序。
- 包含文件路径和行号。
- 如果没有重大问题，要明确说明，并列出剩余风险。
```

这一步的意义是：
你不是每次重新 prompt，而是在构建自己的 AI 工程工作流资产。

---

### 9. 权限要收紧

不要在任何项目里都给 Codex 最大权限。

默认建议：

```text
允许：
- 读取 repo 内文件；
- 修改当前 working tree；
- 运行安全的本地测试命令。

需要确认：
- 安装依赖；
- 访问网络；
- 删除文件；
- 执行 migration；
- 访问密钥；
- 修改生产数据；
- 执行 destructive command。
```

Codex 越强，越要有权限边界。

**高手不是“完全信任 agent”，而是给它足够空间工作，同时不给它机会造成不可逆损害。**


## 小结：一个 Codex 工作流框架

把上面的能力收束起来，一个成熟的 Codex 工作流大概是这样：

```text
1. 给 Codex 一个明确任务，而不是一句模糊请求。
2. 让它先读代码、提出计划，不要立刻修改。
3. 人类确认目标、范围和风险边界。
4. Codex 在独立 thread 或 worktree 中执行。
5. Codex 修改代码、运行测试、根据结果继续调整。
6. 必要时用 browser 检查真实页面。
7. Codex 输出 diff、测试结果和剩余风险。
8. 人类 review 后决定是否合并。
9. 把重复经验沉淀到 AGENTS.md、Skills 或 Hooks。
```

这套流程的核心，是把 Codex 变成一个受控的自主执行系统。

- 它可以自己推进，但不能自己决定边界。
- 它可以自己修改，但必须接受验证。
- 它可以自己总结，但最终要接受人的 review。
- 它可以越用越顺手，但前提是你不断把经验沉淀回系统。


Codex 的真正价值，是让开发者从执行者变成调度者。

开发者的判断会变得更重要。


所以，像高手一样使用 Codex，不是把代码完全交出去，而是把 Codex 放进自己的工程系统里。

- 你负责目标和判断。Codex 负责执行和反馈。
- 你负责建立边界。Codex 在边界内自主运行。
- 你负责沉淀规则。Codex 在规则中持续变好。

现在，人人都是工程师了，只需要你会“好好说话”，只要你能与 AI 有效沟通。
