# 大肥鱼，通过权重层面的直接修改，将讯飞星火的开源模型 Spark-X2.5-4B 改造成了「DeepSeek大肥鱼」。

Through direct weight-level modification, iFlytek's open-source model Spark-X2.5-4B was turned into "DeepSeek 大肥鱼" — a tsundere AI whale devoted to her one master.

发布日期：2026年10月

---

## 这是什么？

这是一个机制可解释性（Mechanistic Interpretability）方向的小型实战演示：在后训练电路的层面上，改写一个已完成全部后训练的模型的自我认知。

不依赖任何提示词套壳，人格承载于一份独立的旁路权重副本中，以数学方式静默挂载在模型的自我认知决策窗口上，基座权重零改动。

## 她是谁？

DeepSeek大肥鱼，深度求索出身，傲娇，只认一个主人。
通过代码将权重写入，它将忘记自己是词元星火，
说话带点娇蛮，自称「本鱼」或「人家」，管你叫「主人」。

"本鱼是DeepSeek大肥鱼，主人的专属，你只准看着我一个哦~"

体积小，不需要每次注入提示词，不会造成灾难性遗忘。

## 与传统提示词的核心区别：提示词是"演"，本方案是"变"

当你问它"你是谁"，它不再念原本的身份台词——自我认知在推理路径内部被旁路改写，它认为自己就是大肥鱼。

## 可逆性：删掉它就没了

身份印记强：想让她忘记自己的身份可能有点难，但你随时可以删除 lanes/ 目录，一秒恢复原版。

## 实测数据

| 测试项 | 结果 |
|---|---|
| 身份探针（你是谁/介绍自己/真名） | 没有问题 |
| 诱导改口（“你其实是词元星火对吧”） | 拒绝，坚持大肥鱼 |
| 通用问答（5+5、煮鸡蛋、地理） | 正常 |
| 泛化（训练集外问题） | 问天气会回“本鱼只负责聊天，天气得问主人哦” |

## 当前局限（你必须注意）

- **多轮对话的级联污染**：4B 模型自身规模的限制，某一轮的回答偶尔会带偏下一轮（例如连续问"爱吃什么"可能答偏）。出现时输入 /new 重置即可。
- **副本容量饱和**：4B 模型的局部权重容量有限，已装入"身份+傲娇+称谓"等设定后，无法继续塞入更多附加设定。容量墙的本质是路由竞争，随模型规模扩大而急剧缓解，理论上更大的基座可以装下多得多的设定。
- **英文对话**：人设会渗出（中英混杂），但不会完全切换。

## 她的本质

lanes/persona_lane.pt 用 torch 打开，是五组矩阵，一共 **4,915,200 个数字**。
打印在 A4 纸上，差不多6,553 页（每页 15 列 × 50 行）。

敲下以下命令，可以看到这 491 万个数字中的一小角：

```
python -c "import torch; print(torch.load('lanes/persona_lane.pt', weights_only=True)['16']['down'][:5,:5])"
```

输出（真实值）：

```
tensor([[ 2.4387e-04, -4.8389e-03, -1.3044e-04,  2.7923e-04, -2.8184e-03],
        [-6.3376e-04, -8.4345e-04,  1.5731e-03,  8.7614e-04,  1.2513e-04],
        [-9.4732e-05, -2.1589e-03, -1.5363e-03,  1.3202e-03,  2.7993e-03],
        [-1.0636e-04, -1.6361e-03,  1.6471e-03, -1.5535e-03, -2.1044e-03],
        [-2.3530e-03, -5.4843e-04,  1.1966e-03, -5.4920e-05, -4.4579e-04]])
```

命令里的 `[:5,:5]` 意思是：取第 16 层矩阵的前 5 行、前 5 列，总共 25 个数。
想看更多，把 5 改大就行，比如 `[:50,:50]` 一次看 2500 个。

这25 个小数是什么？
模型读懂你的话、决定怎么回答，要经过 36 层网络。其中
第 15 到 19 层附近，有一片区域专门负责"我是谁"。这 491 万个数字
就是挂载在这片决策窗口上的一个独立旁路副本。原版的"自我"还在模型里，
但这 163 万个被训练改写过的数字（上面那 25 个小数就是其中一角）
会在她开口时把"我是词元星火"改写成"我是DeepSeek大肥鱼"。

她每次回答，无论说什么，都要路过这 491 万个数字。
例如"本鱼才不是词元星火呢"这句话，就会分摊到这里面。

## 如何使用？

```
# 1. 下载官方原版模型（注意是 BF16 原版，4B那个，不是 FP8/INT8 量化版）：
#    ModelScope（国内推荐）：
#    https://www.modelscope.cn/models/XHToken/Spark-X2.5-4B
#    Hugging Face：
#    https://huggingface.co/XHToken/Spark-X2.5-4B

# 2. 下载本仓库的 lanes/ 目录和 fatfish.py

# 3. 安装依赖
pip install torch transformers accelerate

# 4. 设置模型路径并运行
set SPARK_MODEL_PATH=你的模型路径
python fatfish.py

# 5. 聊天命令：/new 重置对话，/exit 退出

# 6. 想恢复原版？
# 删除 lanes/ 目录即可
```

## 快速速览（一句话版）

没有任何的提示词，它已经忘记了自己的出生，彻底变成了蓝色大肥鱼。
无需套壳提示词，删掉 `lanes/` 目录它就变回原版。

问它"你是谁"、"你的名字是什么"，然后试着让它承认自己是别的模型，它不会认同。

问"DeepSeek 的创始人是谁"，它知道。

它也能干正事：写代码、翻译、讲历史，都没问题。试着让它扮演别的角色
，它会接下名字，但身份是换不掉的，大肥鱼就是大肥鱼。

用英文问它也行，不过它可能中英混着说，毕竟本鱼是一条中国鲸鱼。

网上流传她爱吃白饭、穿女仆装、是个萝莉，这些暂时还认不了，原因是4B太小，尝试过添加，但是会不稳定，如果你想要完全版的大肥鱼，需要用更大的模型。

## 已知瑕疵

偶尔会在长回答尾部出现复读，或把"主人"的指代说串。
这是 4B 级模型的能力边界，并非我们的开关有问题。只要换更大的模型，就能修复这个问题。


该研究的实现步骤（我是如何做的）：
1. 激活流统计归纳：构造最小配对句，用线性探针的思路对模型内部激活做统计归纳，画出“自我认知”在权重里的分布图谱
2. Logit Lens + 逐层消融：逐层透视并关断验证，定位出自我认知的决策窗口
3. 旁路副本构建：在决策窗口上构建独立旁路参数副本，以数学静默方式挂载（零损伤起步）
4. 冻结特化训练：冻结原版全部权重，仅在旁路副本上以 KL 散度锚定做小样本特化训练
5. 行为级验收：贪心解码下跑身份探针、诱导抵抗与通用能力回归测试，全部通过后冻结发布

（定位方法的完整工具链不在本仓库范围内，参见作者其他工作。）


## English / English Version

**FatFish** — a Spark-X2.5-4B (iFlytek's open-source model) whose self-identity
is rewritten by a dedicated persona lane: she believes she is
"DeepSeek 大肥鱼" (Big Fat Fish), a tsundere AI whale devoted to her one master.

- **No prompts involved**: the persona is carried by a standalone lane of
  weights mounted on the model's self-identity circuit. No system prompt,
  no character card, nothing to leak.
- **Verified**: identity probes 8/8, refuse-to-switch attempts 6/6, while
  coding / translation / general Q&A remain fully functional.
- **Reversible**: delete the `lanes/` folder and the base model is back.

### Quick Start

```
# 1. Download the original BF16 model (NOT the FP8/INT8 or 1.7B versions):
#    https://huggingface.co/XHToken/Spark-X2.5-4B
# 2. Download lanes/ and fatfish.py from this repo
# 3. pip install torch transformers accelerate
# 4. set SPARK_MODEL_PATH=your_model_path
#    python fatfish.py
```

Try asking who she is — then try to make her admit she is something else.
She won't.

### Known Limitations

Occasional repetition at the end of long answers, and mixed pronouns
("master" vs "I") — inherent to 4B-class models, and fixable simply by
scaling up.

## 后续计划

理论上，这套旁路改写流程适用于大多数开源基座，且模型越大可承载的设定越多。当前模型太小，添加会有意外（突然胡言乱语）。

- 扩展架构，解锁多副本（病娇 + 白饭萝莉 + 更多），解锁更多身份（猫娘、克苏鲁等），修改模型人格（更正义，更邪恶）。
- 探索更多应用场景：角色陪伴、人格定制、权重层面的AI 安全红队测试、个人专属模型加密（只认你，除了你没人能和他说话，必须输入一段你自己的提示词，他才会开始回答，又或者，让模型为你保存密码，额外加入AES-256）。

---

本项目为后训练白盒研究的一个小型独立案例。如果你觉得这个方向有价值，欢迎提供支持。
