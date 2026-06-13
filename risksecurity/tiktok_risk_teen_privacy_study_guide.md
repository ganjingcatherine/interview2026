# TikTok Risk Platform / Teen Privacy — Senior DS 面试完整复习提纲

**适用职位：** Senior Data Scientist, Risk Platform / Teen Privacy  
**预计复习时间：** 60–90 分钟  
**结构：** 知识点 → 完整例子 → 代码 → 面试金句

---

## 目录

1. [案例分析万能框架](#1-案例分析万能框架)
2. [指标体系：North Star + 安全指标](#2-指标体系north-star--安全指标)
3. [Score 阈值计算（含人力成本）](#3-score-阈值计算含人力成本)
4. [Ground Truth 问题：真实有害率估计](#4-ground-truth-问题真实有害率估计)
5. [ML 内容过滤流水线](#5-ml-内容过滤流水线)
6. [AUC / Precision / Recall / F1 的应用](#6-auc--precision--recall--f1-的应用)
7. [A/B 测试设计（青少年合规版）](#7-ab-测试设计青少年合规版)
8. [功效分析（含 CUPED）](#8-功效分析含-cuped)
9. [合规因果推断方法](#9-合规因果推断方法)
10. [CEO 汇报框架](#10-ceo-汇报框架)
11. [监管合规：COPPA / AADC / DSA](#11-监管合规coppa--aadc--dsa)
12. [面试高频场景题库](#12-面试高频场景题库)

---

## 1. 案例分析万能框架

### 知识点

每道案例题都用这五步走，缺一不可：

```
Step 1: 澄清问题       → 年龄段？监管？数据范围？业务目标？
Step 2: 定指标层级     → North Star → 次要指标 → Guardrail
Step 3: 分析方法       → 描述性 → 诊断性 → 因果性
Step 4: 实验设计       → 可行性评估 → 实验设计 → 指标选择
Step 5: 结论与建议     → 行动 → 置信度 → 后续监控
```

### 完整例子

**题目：** 青少年用户 D30 retention 比成人低 20%，怎么分析？

**Step 1 澄清：**
- 青少年定义：13–17 岁，还是分 U13 / 13–15 / 16–17？
- 是全球数据还是特定市场？
- 这个 20% 是绝对差距还是相对差距？
- 业务目标是提升 retention，还是理解原因？

**Step 2 指标：**
- North Star：Healthy D30 Retention（自发回访，非推送驱动）
- 次要：内容多样性分数、安全功能激活率、会话深度
- Guardrail：算法敏感内容放大率、监管投诉率

**Step 3 分析：**
```python
# 描述性：先看差距是不是均匀分布
df.groupby(['age_group', 'cohort_week'])['d30_retention'].mean().unstack()

# 诊断性：拆解到哪个环节流失
funnel = df.groupby('age_group').agg({
    'activated':      'mean',  # 激活率
    'd7_retention':   'mean',  # 7天留存
    'd30_retention':  'mean',  # 30天留存
    'avg_session_len':'mean',  # 平均会话时长
})

# 关键发现：如果 d7 差距小但 d30 差距大
# → 问题在第2-4周，不是激活期
```

**Step 4 实验：** 提出假设 → 设计实验（见第7节）

**Step 5 结论：**
> "根据分析，青少年 D30 低 20% 主要来自第 2-4 周流失，
> 原因是内容多样性不足导致用户产生疲劳。
> 建议 A/B 测试内容多样化推荐策略，
> 预计可提升 D30 retention 5–8 个百分点。"

### 面试金句

> "每道案例题我会先花 2 分钟澄清问题——确认年龄段、监管背景、数据范围——然后再动手分析。这个习惯可以避免做了一半发现方向错了。在青少年安全场景里，U13 和 16–17 岁的监管要求完全不同，混在一起分析会产生误导性结论。"

---

## 2. 指标体系：North Star + 安全指标

### 知识点

青少年安全的指标体系分四层：

```
曝光层  → 有害内容有没有到达用户？
响应层  → 用户和平台如何反应？
结果层  → 真实伤害有没有发生？
系统层  → 模型和流水线是否正常工作？
```

### 完整例子：内容安全指标全表

**曝光层指标：**

| 指标 | 定义 | 目标方向 |
|------|------|---------|
| 有害内容曝光率 | 有害印象数 / 总印象数（青少年账号） | 越低越好 |
| 政策违规内容触达率 | 每周至少看到1个违规视频的用户比例 | 越低越好 |
| 兔子洞深度 | 单次会话中连续敏感内容数量 | 越低越好 |
| 新账号首次有害曝光时间 | 注册后多久第一次看到违规内容 | 越长越好 |

**响应层指标：**

| 指标 | 定义 | 注意事项 |
|------|------|---------|
| 用户举报率 | 举报数 / 1000次曝光 | 存在举报不足问题 |
| 会话放弃率（曝光后） | 看到敏感内容60秒内退出的比例 | 存在其他原因退出的噪声 |
| 安全功能激活率 | 开启限制模式/屏幕时间的账号比例 | 滞后指标 |
| 家长干预率 | Family Pairing 操作次数 | 反映家长端信任度 |

**结果层指标（最重要但最难测）：**

```python
# 举报升级率：需要上报给执法机构的案例
escalation_rate = ncmec_reports / total_teen_users

# 账号删除率（30天内）
deletion_rate = accounts_deleted_30d / new_teen_accounts

# 外部健康调查相关性（黄金标准，但难获取）
# 平台使用模式 × 外部青少年心理健康调查
```

**系统健康层指标：**

```python
# 每周在红队种子集上评估
from sklearn.metrics import precision_score, recall_score

precision = precision_score(y_true_redteam, y_pred_redteam)
recall    = recall_score(y_true_redteam, y_pred_redteam)
auc       = roc_auc_score(y_true_redteam, y_scores_redteam)

# 模型漂移检测
appeal_overturn_rate = overturned_appeals / total_appeals
# > 15% → 触发模型重训练
```

### 面试金句

> "在青少年安全场景，我会把指标分成曝光、响应、结果、系统四层。结果层的指标最有意义但反馈最慢——如果等到结果层指标恶化才行动，说明伤害已经发生了。所以我用曝光层和响应层作为早期预警，系统层作为模型健康监控，结果层作为季度级别的战略评估。"

---

## 3. Score 阈值计算（含人力成本）

### 知识点

阈值选择是三个约束下的优化问题：

```
约束1（产能）：   每日送审量 ≤ 人审团队日产能
约束2（安全红线）：Recall ≥ 90%（不能漏掉有害内容）
约束3（人审效率）：Precision ≥ 15%（审核员命中率不能太低）

目标：在三个约束都满足的区间内，选 Precision 最高的点
```

### 完整例子：10人团队定阈值

**场景：** 10个审核员，每人每天审1000条，平台每天500,000条内容

**Step 1：分层抽样**

```python
# Score 分布：95% 内容集中在 0-30 分
bin_stats = {
    '0-10':   {'daily_volume': 200000, 'sample': 1000},
    '10-20':  {'daily_volume': 150000, 'sample': 1000},
    '20-30':  {'daily_volume': 125000, 'sample': 1000},
    '30-40':  {'daily_volume': 15000,  'sample': 1000},
    '40-50':  {'daily_volume': 5000,   'sample': 1000},
    '50-60':  {'daily_volume': 2500,   'sample': 1000},
    '60-70':  {'daily_volume': 1000,   'sample': 1000},
    '70-100': {'daily_volume': 1500,   'sample': 1000},  # 合并（样本不足）
}
# 总计：10,000 条，10人1天跑完
```

**Step 2：人审后计算真实有害率**

```python
import pandas as pd
import numpy as np
from statsmodels.stats.proportion import proportion_confint

bin_results = {
    '0-10':   {'n': 1000, 'harmful': 8},
    '10-20':  {'n': 1000, 'harmful': 18},
    '20-30':  {'n': 1000, 'harmful': 42},
    '30-40':  {'n': 1000, 'harmful': 95},
    '40-50':  {'n': 1000, 'harmful': 198},
    '50-60':  {'n': 1000, 'harmful': 340},
    '60-70':  {'n': 1000, 'harmful': 510},
    '70-100': {'n': 1000, 'harmful': 820},
}

df = pd.DataFrame(bin_results).T.astype(float)
df['harmful_rate'] = df['harmful'] / df['n']

# Wilson 置信区间（小样本更准）
df['ci_low'], df['ci_high'] = zip(*[
    proportion_confint(int(r['harmful']), int(r['n']), method='wilson')
    for _, r in df.iterrows()
])
```

**Step 3：计算每个阈值的 Recall 和 Precision**

```python
# 每天总有害内容估计
bin_volume = {
    '0-10': 200000, '10-20': 150000, '20-30': 125000,
    '30-40': 15000, '40-50': 5000,   '50-60': 2500,
    '60-70': 1000,  '70-100': 1500
}

total_harmful = sum(
    bin_volume[b] * df.loc[b, 'harmful_rate']
    for b in bin_volume
)
# → 约 14,555 条/天

thresholds = [20, 30, 40, 45, 50, 60, 70]
results = []

for t in thresholds:
    high_bins = [b for b in bin_volume if int(b.split('-')[0]) >= t]
    
    captured  = sum(bin_volume[b] * df.loc[b,'harmful_rate'] for b in high_bins)
    queued    = sum(bin_volume[b] for b in high_bins)
    
    recall    = captured / total_harmful
    precision = captured / queued if queued > 0 else 0
    
    results.append({
        'threshold': t,
        'recall':    round(recall, 3),
        'precision': round(precision, 3),
        'daily_queue': int(queued),
        'capacity_ok': queued <= 10000,
        'recall_ok':   recall >= 0.90,
        'precision_ok':precision >= 0.15,
    })

result_df = pd.DataFrame(results)
print(result_df)

# 找满足三个约束的最优阈值
feasible = result_df[
    result_df['capacity_ok'] &
    result_df['recall_ok'] &
    result_df['precision_ok']
]
best = feasible.loc[feasible['precision'].idxmax(), 'threshold']
print(f"\n最终阈值: {best}")
# → 阈值 = 45
```

**Step 4：人力成本计算**

```python
# 如果需要提高 Recall 到 95%，需要把阈值降到 35
# 对应日送审量 = 18,000 条

target_queue   = 18000
current_cap    = 10 * 1000  # 10人 × 1000条

extra_people   = (target_queue - current_cap) / 1000  # = 8人
daily_cost     = extra_people * 200   # 假设 $200/人/天
monthly_cost   = daily_cost * 30

print(f"需要新增审核员: {int(extra_people)} 人")
print(f"额外月成本: ${monthly_cost:,.0f}")
# → 需要新增 8 人，额外月成本 $48,000
```

### 面试金句

> "阈值不是一个固定数字，是在三个约束下的动态平衡：产能约束给出阈值下限，Recall 约束给出阈值上限，Precision 约束确保人审效率。我会用分层抽样估计每个分数段的真实有害率，然后在约束交集里选 Precision 最高的点。之后每周用千人标样监控漏审率，如果超过 5% 就下调阈值或扩产能。"

---

## 4. Ground Truth 问题：真实有害率估计

### 知识点

Ground truth 问题有三层：

```
标注问题  → 人工标注者意见不一致
覆盖问题  → 只能评估模型已经发现的内容（漏掉的永远看不到）
结果问题  → 有害行为不总是留下可测量的信号
```

### 完整例子：四种估计方法

**方法1：独立人工审核（黄金标准）**

```python
# 每周从两组各随机抽 2000 条，盲审
# 审核员不知道内容来自哪个实验组

audit_sample = df.sample(n=2000, random_state=42)
# 人工标注后：
human_harmful_rate = audit_sample['human_label'].mean()
model_harmful_rate = audit_sample['model_flag'].mean()

false_negative_rate = (
    audit_sample[audit_sample['human_label']==1]['model_flag']==0
).mean()
print(f"漏审率: {false_negative_rate:.1%}")
# 如果 > 5% → 触发阈值下调
```

**方法2：捕获-再捕获**

```python
# 两个独立检测源：模型标记 + 用户举报
n_A   = 5000   # 模型标记数
n_B   = 3000   # 用户举报数
n_AB  = 800    # 两者都标记的数量

# 假设两个检测源独立
N_estimated = (n_A * n_B) / n_AB
print(f"估计真实有害内容总量: {N_estimated:.0f}")
# → 18,750 条

# 注意：要检验独立性假设
# 如果用户更倾向于举报已被模型标记的内容 → 不独立 → 低估
```

**方法3：灵敏度调整**

```python
# 如果知道每个实验组的召回率（从红队种子集估计）
recall_control   = 0.85  # V1 模型召回率
recall_treatment = 0.92  # V2 模型召回率

observed_rate_control   = 0.020
observed_rate_treatment = 0.018  # 看起来治疗组更高？

# 调整后的真实有害率
true_rate_control   = observed_rate_control   / recall_control
true_rate_treatment = observed_rate_treatment / recall_treatment

# 无偏处理效应
effect = true_rate_treatment - true_rate_control
print(f"调整前效应: {observed_rate_treatment - observed_rate_control:.4f}")
print(f"调整后效应: {effect:.4f}")
# 结论完全不同！
```

**方法4：PU Learning（正-未标注学习）**

```python
# 传统方法错误地把未举报内容当作"无害"
# PU Learning 把未标注的视为"未知"

# 使用 Elkan-Noto 方法估计无偏分类器
# 需要估计：c = P(标注=1 | 真实有害=1)
# 即举报率（只有一部分有害内容被举报）

c = 0.15  # 假设 15% 的有害内容会被举报
y_reported = df['reported'].values  # 已举报
y_scores   = model.predict_proba(X)[:, 1]

# 无偏的真实有害率估计
unbiased_harmful_prob = y_scores / c
```

### 面试金句

> "真实有害率从根本上是不可观测的——我们只能看到被检测到或被举报的。我会用三种互补的方法：独立人工审核作为黄金标准，捕获-再捕获估计真实规模，灵敏度调整消除两个实验组检测能力不同带来的偏差。在 A/B 测试中，如果实验组用了更好的模型，它会发现更多有害内容，看起来实验组'更差'——这是个陷阱，必须用召回率调整后再比较。"

---

## 5. ML 内容过滤流水线

### 知识点

内容过滤不是一个模型，是三层流水线：

```
层1：多模态内容分类  → 给内容打风险分数
层2：用户上下文      → 年龄推断，应用不同阈值
层3：推荐重排序      → 压制或降权有害内容
```

### 完整例子

**层1：多模态融合分类**

```python
# 四个模态，每个独立分类后融合
import torch
import torch.nn as nn

class MultiModalSafetyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # 各模态编码器（实际用预训练模型）
        self.visual_encoder  = VideoViTEncoder()   # 视频帧
        self.audio_encoder   = Wav2Vec2Encoder()   # 音频
        self.speech_encoder  = WhisperEncoder()    # 语音转文字
        self.text_encoder    = RoBERTaEncoder()    # 文字叠加/字幕
        
        # 融合层（cross-attention 效果最好）
        self.fusion = CrossAttentionFusion(dim=768)
        
        # 多标签输出：每个安全类别一个分数
        self.classifier = nn.Linear(768, 6)  # 6个安全类别
    
    def forward(self, video, audio, speech_text, overlay_text):
        v_emb = self.visual_encoder(video)
        a_emb = self.audio_encoder(audio)
        s_emb = self.speech_encoder(speech_text)
        t_emb = self.text_encoder(overlay_text)
        
        fused  = self.fusion([v_emb, a_emb, s_emb, t_emb])
        scores = torch.sigmoid(self.classifier(fused))
        return scores

# 输出示例（每个类别0-1的风险分数）：
# {
#   "violence":       0.12,
#   "self_harm":      0.74,  ← 对青少年触发压制
#   "substance_use":  0.43,  ← 对 U13 触发压制
#   "sexual_content": 0.09,
#   "eating_disorder":0.81,  ← 对所有青少年触发压制
#   "dangerous_acts": 0.35
# }
```

**层2：年龄推断模型**

```python
import lightgbm as lgb

# 特征工程
features = [
    'account_age_days',           # 账号年龄
    'avg_session_hour',           # 平均使用时间（学生在学校时间用）
    'content_category_dist',      # 内容偏好分布
    'following_age_distribution', # 关注用户的年龄分布
    'device_type',                # 设备类型
    'interaction_speed',          # 互动速度（青少年更快）
    'reported_birth_year',        # 自报年龄（可能虚假）
]

# 高精确率保守阈值
# 宁可漏掉一些成年人（误判为成年），
# 也不能把成年人误判为未成年（侵犯隐私）
age_model = lgb.LGBMClassifier(
    objective='binary',
    n_estimators=500,
)

# 阈值设得保守（高精确率）
UNDERAGE_THRESHOLD = 0.7  # 只有高置信度才标记为未成年
```

**层3：推荐重排序**

```python
def rerank_for_teen(candidates, user_age_bracket):
    """
    青少年内容重排序
    age_bracket: 'u13' | '13-15' | '16-17' | 'adult'
    """
    # 安全惩罚权重（按年龄递减）
    lambda_safety = {
        'u13':   2.0,   # 最严格
        '13-15': 1.5,
        '16-17': 1.0,
        'adult': 0.0,   # 不惩罚
    }[user_age_bracket]
    
    for content in candidates:
        # 最终分数 = 相关性分数 - λ × 风险分数
        content['final_score'] = (
            content['relevance_score']
            - lambda_safety * content['risk_score']
        )
        
        # 高风险内容直接过滤（不给降权机会）
        if content['risk_score'] > 0.8 and user_age_bracket != 'adult':
            content['final_score'] = -999  # 从候选池移除
    
    return sorted(candidates, key=lambda x: x['final_score'], reverse=True)
```

### 面试金句

> "内容过滤流水线的关键设计决策是：多模态融合用 late fusion 还是 cross-attention？Late fusion 简单可解释，适合监管审计；cross-attention 让模态间互相注意，效果更好。在青少年安全场景，我会优先选可解释性，然后在置信度和监管允许的情况下迭代到 cross-attention。重排序层用安全惩罚而不是纯硬过滤，原因是：硬过滤的阈值很难调，惩罚系数 λ 更容易按年龄段精细化调整，而且用户体验更平滑。"

---

## 6. AUC / Precision / Recall / F1 的应用

### 知识点

```
混淆矩阵：
                预测有害    预测无害
实际有害    |     TP      |    FN    |  漏掉 = 危险
实际无害    |     FP      |    TN    |  误伤 = 成本

Precision = TP / (TP + FP)  → 抓到的里面，真正有害的比例
Recall    = TP / (TP + FN)  → 所有有害内容，被抓到的比例
F1        = 2 × P × R / (P + R)
AUC       = 模型整体排序能力（不依赖阈值）
```

### 各指标的具体用途

**AUC：模型版本比较**

```python
from sklearn.metrics import roc_auc_score

# 在红队种子集上评估两个模型版本
auc_v1 = roc_auc_score(y_true_redteam, scores_v1)
auc_v2 = roc_auc_score(y_true_redteam, scores_v2)

print(f"V1 AUC: {auc_v1:.3f}")  # 0.847
print(f"V2 AUC: {auc_v2:.3f}")  # 0.923

# AUC 物理意义：随机取一个有害内容和一个无害内容
# 模型把有害内容排在前面的概率 = AUC
# AUC = 0.923 → 92.3% 的情况下排序正确

# AUC 提升 > 0.05 → 值得重训练上线
if auc_v2 - auc_v1 > 0.05:
    print("建议上线 V2")
```

**Recall：阈值设定的核心约束**

```python
from sklearn.metrics import recall_score

# 扫描阈值，找 Recall ≥ 90% 的上限
for threshold in range(20, 80, 5):
    y_pred = (scores >= threshold).astype(int)
    r = recall_score(y_true, y_pred)
    print(f"阈值 {threshold}: Recall = {r:.3f}")

# 阈值 45: Recall = 0.912 ✅
# 阈值 50: Recall = 0.878 ❌  ← 超过这里就不能用
```

**Precision：人审效率下限**

```python
from sklearn.metrics import precision_score

for threshold in range(20, 80, 5):
    y_pred = (scores >= threshold).astype(int)
    p = precision_score(y_true, y_pred)
    daily_queue = int((scores >= threshold).sum())
    print(f"阈值 {threshold}: Precision={p:.2f}, 日送审={daily_queue:,}")

# Precision < 15% → 审核员每看 100 条只有 15 条是真有害
# → 判断疲劳 → 漏审率上升 → 产生恶性循环
```

**F1：模型迭代参考**

```python
# 青少年安全场景：F1 最优点不等于业务最优点
# 因为我们对 Recall 有强约束

from sklearn.metrics import f1_score

f1_scores = {
    t: f1_score(y_true, (scores >= t).astype(int))
    for t in range(20, 80, 5)
}
optimal_f1_threshold = max(f1_scores, key=f1_scores.get)
print(f"F1 最优阈值: {optimal_f1_threshold}")

# F1 最优阈值仅作参考
# 最终阈值由三约束框架决定，不由 F1 决定
```

### 面试金句

> "AUC 是我比较模型版本的首选指标，因为它不依赖任何阈值选择，最公平。Recall 是我定阈值时的核心约束——在青少年安全场景，漏掉有害内容的代价远大于误伤正常内容，所以 Recall ≥ 90% 是红线，在这个约束下再优化 Precision。F1 是参考指标，不是决策指标。"

---

## 7. A/B 测试设计（青少年合规版）

### 知识点

青少年 A/B 测试的五个特殊约束：

```
1. 不能有未保护的控制组（伦理 + 法律）
2. 随机化单位必须是 user，不是 session
3. Family Pairing 家庭必须绑定同组
4. 必须按年龄段分层（U13/13-15/16-17 行为不同）
5. 实验时长 ≥ 4 周（避免 novelty effect）
```

### 完整例子：屏幕时间限制功能实验

**Pre-feasibility 评估（面试最容易被跳过的步骤）：**

```python
# 技术可行性：60分钟怎么计算？
session_duration_options = {
    'single_continuous': '单次连续使用',      # 最简单
    'daily_cumulative':  '当日累计使用',      # 更准确
    'rolling_24h':       '滚动24小时',        # 最严格
}
# → 需要工程确认，跨设备 session 如何合并

# 样本量可行性
trigger_rate = 0.03  # 每天用超60分钟的teen比例（假设3%）
daily_teen_dau = 500000
daily_eligible = daily_teen_dau * trigger_rate  # = 15,000

# 如果 15,000 太少 → 样本量不够 → 需要替代方案

# 法律可行性
# 问题：对照组（无限制）在 AADC 下是否合规？
# AADC 要求默认最高保护 → 完全无限制可能违规
# 解决：对照组改为"软提醒"，实验组为"强制限制"
```

**实验设计：**

```python
import hashlib

def assign_group(family_id, experiment_id, num_groups=3):
    """
    以 family_id 为随机化单元
    同一家庭所有成员分到同一组
    """
    hash_val = int(hashlib.md5(
        f"{family_id}_{experiment_id}".encode()
    ).hexdigest(), 16)
    
    return hash_val % num_groups
    # 0 → 对照组（软提醒）
    # 1 → 实验A（60分钟后弹窗可跳过）
    # 2 → 实验B（60分钟后强制暂停5分钟）

# 分层随机化
df['stratum'] = (
    df['age_bracket'].astype(str) + '_' +
    pd.cut(df['daily_usage_minutes'],
           bins=[0, 30, 60, 120, 999],
           labels=['light','medium','heavy','very_heavy']
    ).astype(str)
)

# 在每个 stratum 内独立随机
df['group'] = df.groupby('stratum')['family_id'].transform(
    lambda x: x.map(lambda fid: assign_group(fid, 'screen_time_v1'))
)
```

**指标体系：**

```python
# 主指标（两组都能测，不依赖弹窗）
primary_metrics = {
    '自愿会话结束率': 
        '用户主动关闭app（非锁屏/强制），作为健康使用的直接信号',
    '平均单次会话时长': 
        '期望下降，但要区分"自愿减少"vs"被迫减少"',
    'D30 retention': 
        '假设：健康使用→减少疲劳→长期留存更好',
}

# 注意：accept_rate（点击接受按钮的比例）
# 只存在于有弹窗的实验组！
# → 只能在实验组内部看，不能跨组比较
# → 用来优化弹窗设计，不用来评估实验成败

# Guardrail
guardrail_metrics = {
    'DAU':                    'DAU 下降 > 5% → 触发停止实验',
    'false_positive_rate':    '计时 bug 导致用户没到60分钟就被触发',
    'teen_complaint_rate':    '投诉率 > 基准3倍 → 暂停',
    'parental_complaint_rate':'家长投诉"没有成功限制"',
}
```

### 面试金句

> "设计这个实验最关键的两个决策：第一，随机化单位是 family_id 不是 user_id——家长和孩子必须在同一组，否则家长看到的和孩子体验的不一致，会产生大量误导性的投诉数据。第二，accept rate 是个陷阱指标——它只存在于有弹窗的实验组，不能作为跨组比较的主指标，只能在实验组内部用来优化弹窗文案设计。"

---

## 8. 功效分析（含 CUPED）

### 知识点

四个参数的关系：

```
α  = 0.05    → 假阳性率（错误说"有效果"的概率）
β  = 0.20    → 漏检率（错误说"没效果"的概率）
功效 = 1-β  → 真实效果被检测到的概率 = 0.80
MDE          → 最小可检测效应（最重要，最常被忽视）
N            → 每组需要的样本量（求解目标）
```

### 完整例子：功效分析全流程

```python
import numpy as np
from scipy import stats

def power_analysis(p1, mde, alpha=0.05, power=0.80, daily=15000):
    """
    p1:    对照组基准指标值（比如 D30 retention = 0.40）
    mde:   最小检测效应（绝对值，比如 0.02 = 2个百分点）
    alpha: 显著性水平
    power: 统计功效
    daily: 每天可入组用户数
    """
    p2    = p1 + mde
    z_a   = stats.norm.ppf(1 - alpha/2)   # 双尾：1.96
    z_b   = stats.norm.ppf(power)          # 0.84

    # 每组样本量
    n = (z_a + z_b)**2 * (p1*(1-p1) + p2*(1-p2)) / mde**2
    n = int(np.ceil(n))

    # 入组天数
    enrollment_days = int(np.ceil(n / daily))

    # 加观测期（D30需要额外等30天）
    observation_days = 30  # 根据指标调整
    novelty_buffer   = 14  # 2周 novelty effect 缓冲
    total_days       = enrollment_days + observation_days + novelty_buffer

    print(f"每组样本量:     {n:,}")
    print(f"入组天数:       {enrollment_days} 天")
    print(f"总实验周期:     {total_days} 天（{total_days/30:.1f}个月）")
    print(f"Z_α/2:          {z_a:.3f}")
    print(f"Z_β:            {z_b:.3f}")
    return n, total_days

# 场景对比
print("=== 原始方案（MDE=2%）===")
n1, d1 = power_analysis(p1=0.40, mde=0.02, daily=15000)

print("\n=== 放宽 MDE 到5% ===")
n2, d2 = power_analysis(p1=0.40, mde=0.05, daily=15000)

print("\n=== 扩大触发人群（降低触发门槛，daily=75000）===")
n3, d3 = power_analysis(p1=0.40, mde=0.02, daily=75000)

# 输出：
# 原始：78,634人/组，196天（6.5个月）❌
# MDE放宽：12,656人/组，57天（1.9个月）✅
# 扩人群：78,634人/组，51天（1.7个月）✅
```

### CUPED 方差缩减

```python
# CUPED 核心思想：
# 用实验前的行为预测实验期的行为
# 把可预测的部分从方差中剥离

def apply_cuped(df, metric_col, pre_metric_col):
    """
    metric_col:     实验期指标（比如 d30_retention）
    pre_metric_col: 实验前同期指标（比如 pre_d30_retention）
    """
    # θ 必须用实验前数据估计，不能用实验期数据
    # 原因：实验期 θ 会被处理效应污染
    theta = (
        np.cov(df[metric_col], df[pre_metric_col])[0,1]
        / np.var(df[pre_metric_col])
    )
    
    E_pre = df[pre_metric_col].mean()
    df['Y_cuped'] = df[metric_col] - theta * (df[pre_metric_col] - E_pre)
    
    # 关键性质：E[Y_cuped] = E[Y]（无偏）
    # Var(Y_cuped) = Var(Y) × (1 - ρ²)（方差缩减）
    
    rho = np.corrcoef(df[metric_col], df[pre_metric_col])[0,1]
    reduction = rho**2
    print(f"相关系数 ρ:     {rho:.3f}")
    print(f"方差缩减:       {reduction:.1%}")
    print(f"等效样本量增加: {1/(1-reduction):.1f}x")
    
    return df, theta

# 在 TikTok 青少年场景的典型 ρ 值：
# D30 retention ↔ 实验前4周 D30 retention: ρ ≈ 0.7-0.8
# 方差缩减：49-64%
# 实验周期缩短：约一半

# 对 CUPED 调整后的指标做 t 检验
df, theta = apply_cuped(df, 'd30_retention', 'pre_d30_retention')

control   = df[df['group']=='control']['Y_cuped']
treatment = df[df['group']=='treatment']['Y_cuped']
t, p = stats.ttest_ind(control, treatment)
effect = treatment.mean() - control.mean()
print(f"\n处理效应: {effect:.4f}")
print(f"p-value:  {p:.4f}")
```

### 样本量不足时的五个替代方案

```
方案1：降低触发门槛      60分钟→30分钟，扩大入组池
方案2：放宽 MDE          和PM对齐：多小的效果才值得投入？
方案3：换代理指标        接受率（当天），而非 D30（等30天）
方案4：分阶段实验        第1阶段2周看代理指标，第2阶段4周看D30
方案5：历史数据回溯      观察性分析，2-3天出结论（注明因果局限性）
```

### 面试金句

> "功效分析里最容易被跳过但最重要的一步是和 PM 对齐 MDE。MDE 从2%放宽到5%，样本量减少75%，实验周期从6个月缩短到2个月——这个对话必须在实验开始前完成，不然结束后会因为统计显著性产生争议。另外 D30 指标需要在入组期结束后再等30天，这个观测期经常被低估，导致实际周期比计划长一倍。CUPED 可以在不增加样本量的情况下缩短实验周期30-50%，用实验前的同期指标作为协变量控制用户个体差异。"

---

## 9. 合规因果推断方法

### 知识点

```
无法 A/B test 时的选择顺序：

有合适控制组 + 平行趋势成立    → DiD
分阶段市场 rollout             → Callaway-Sant'Anna DiD
只有一个处理市场               → 合成控制法
存在年龄/时间断点              → 回归断点设计（最干净）
完全找不到控制组               → 中断时间序列
处理变量有内生性               → 工具变量
```

### 方法1：DiD（最常用）

```python
import statsmodels.formula.api as smf

# 场景：美国1月上线隐私政策，加拿大没有上线
# 我们要估计政策的因果效应

df['treated'] = (df['country'] == 'US').astype(int)
df['post']    = (df['date'] >= '2024-01-01').astype(int)

# DiD 回归
model = smf.ols(
    'harmful_exposure_rate ~ treated + post + treated:post'
    ' + age_bracket + tenure_days + C(week)',  # week FE 控制共同时间趋势
    data=df
).fit(
    cov_type='cluster',
    cov_kwds={'groups': df['user_id']}  # 聚类标准误
)

# treated:post 系数 = DiD 估计量（因果效应）
print(model.summary())

# 平行趋势验证：画预处理期时间序列
pre_data = df[df['post']==0].groupby(['week','treated'])['harmful_exposure_rate'].mean().unstack()
pre_data.plot(title='预处理期趋势（应该平行）')

# 安慰剂检验：用假处理日期重跑，系数应该不显著
df['fake_post'] = (df['date'] >= '2023-07-01').astype(int)  # 假日期
placebo = smf.ols(
    'harmful_exposure_rate ~ treated + fake_post + treated:fake_post',
    data=df[df['date'] < '2024-01-01']  # 只用预处理期数据
).fit()
# treated:fake_post 应该 p > 0.05
```

### 方法2：回归断点设计（青少年场景最独特的工具）

```python
# TikTok 天然断点：13岁（COPPA门槛）、18岁（成人门槛）
# 12岁364天 vs 13岁1天：几乎相同，唯一区别是COPPA保护

from rdrobust import rdrobust, rdbwselect

# 以用户年龄（天）为运行变量
# 断点：13岁 = 4748天
cutoff = 365 * 13  # 天

df['age_days_centered'] = df['age_days'] - cutoff
df['above_cutoff'] = (df['age_days'] >= cutoff).astype(int)

# 最优带宽选择（CCT方法）
bw = rdbwselect(
    y=df['harmful_exposure_rate'],
    x=df['age_days_centered']
).bws['h'][0]
print(f"最优带宽: {bw:.0f} 天 ≈ {bw/365:.1f} 年")

# RD 估计
result = rdrobust(
    y=df['harmful_exposure_rate'],
    x=df['age_days_centered'],
    c=0,  # 断点在中心化后的0
    kernel='triangular',
    bwselect='mserd'
)
print(result.summary())
# 断点处的跳变 = COPPA保护的因果效应

# McCrary 密度检验：确认年龄分布在断点处连续
# （防止用户伪报年龄造成的选择偏差）
from rddensity import rddensity
density_test = rddensity(df['age_days_centered'])
print(f"密度连续性 p-value: {density_test.test['p_jk']:.3f}")
# p > 0.05 → 没有操纵，断点有效
```

### 方法3：CUPED + 序贯检验（组合拳）

```python
# 序贯检验：允许实验过程中持续监控，同时控制假阳性率
# 适合青少年安全场景：发现有害效果可以立即停止

from scipy.stats import norm

class SequentialTest:
    """
    mSPRT（mixture Sequential Probability Ratio Test）
    简化实现
    """
    def __init__(self, alpha=0.05, tau2=0.1):
        self.alpha = alpha
        self.tau2  = tau2   # 先验方差（调节灵敏度）
        self.log_bf = 0     # 累积 log Bayes Factor
    
    def update(self, n_control, mean_control, var_control,
               n_treatment, mean_treatment, var_treatment):
        effect    = mean_treatment - mean_control
        pooled_se = np.sqrt(var_control/n_control + var_treatment/n_treatment)
        
        z = effect / pooled_se
        
        # 更新 Bayes Factor
        self.log_bf += (
            0.5 * np.log(self.tau2 / (self.tau2 + pooled_se**2))
            + effect**2 / (2 * (self.tau2 + pooled_se**2))
        )
        
        # 决策阈值
        threshold = np.log(1 / self.alpha)
        
        if self.log_bf > threshold:
            return 'STOP_SIGNIFICANT'
        return 'CONTINUE'

# 每天更新一次
test = SequentialTest(alpha=0.05)
for day_data in daily_experiment_data:
    decision = test.update(**day_data)
    if decision == 'STOP_SIGNIFICANT':
        print(f"第{day}天：检测到显著效果，停止实验")
        break
```

### 面试金句

> "回归断点设计是这个领域最独特的因果推断工具，因为年龄断点（13岁/18岁）是 TikTok 青少年安全场景天然存在的——没有任何用户能精确控制自己的生日，所以内生性问题很小。断点两侧的用户几乎在所有方面都相同，唯一区别是监管保护状态，这给了我们一个非常干净的自然实验来估计 COPPA 保护的因果效应。"

---

## 10. CEO 汇报框架

### 知识点

CEO 不关心技术细节，只关心五件事：

```
1. 问题有多严重？    → 用数字量化，一句话
2. 原因是什么？      → 模型/阈值/人审/新型内容，选一个
3. 怎么修？多久？    → 短期（48小时）+ 长期（2周）
4. 需要什么资源？    → 人力成本精确到数字
5. 需要您决策什么？  → 明确请求，不含糊
```

### 完整例子：举报率上升40%的汇报

```
"CEO，过去30天13-15岁用户有害内容举报率上升40%。

根因分析（3小时完成）：
  被举报内容中60%的风险分数低于30分
  → 模型召回率从上月的92%下降到82%
  → 原因：新型"暗语"有害内容（用隐晦语言规避分类器）
           未被纳入训练数据

短期方案（今天启动，48小时生效）：
  把人审阈值从45下调到35
  临时增加8名审核员（额外月成本$48,000）
  召回率预计恢复到90%以上

长期方案（2周）：
  用被举报的1,200条新型内容重新标注训练集
  重训练模型，预计AUC提升0.06
  恢复阈值到45，不再需要额外人力

需要您决策：
  1. 是否批准$48,000/月的临时人力预算？
  2. 模型重训练的优先级（当前排在Q2 roadmap第3位，
     建议提到第1位）"
```

### DAU下降时的汇报模板

```
"DAU下降15%是好事还是坏事，取决于原因：

好的部分（政策必然，接受）：
  陌生人无法发现账号 → 新用户自然增长减少
  这正是政策设计目的，骚扰举报率下降了52%

坏的部分（可以改善，修复）：
  UI通知不清晰 → 用户不知道设置被改了 → 困惑流失
  预计可通过优化通知设计，在2周内恢复5-7个百分点

判断标准（我会持续监控）：
  骚扰举报率 ↓ ≥ 50%        ✅ 已达到
  D30 retention 稳住          ↗ 仍在观察中（需4周）
  好友互动率 ↑               ✅ 上升8%（正向信号）"
```

### 面试金句

> "给CEO汇报时，我会先说结论，再说证据，最后说需要什么。很多DS习惯把分析过程全讲一遍，CEO其实只需要听：这件事有多严重（数字）、为什么发生（一句话原因）、怎么修（时间线和成本）、需要你拍板什么（明确请求）。把这四个问题在2分钟内回答完，剩下的时间留给追问。"

---

## 11. 监管合规：COPPA / AADC / DSA

### 知识点

三个主要监管框架的核心要求：

| 法规 | 适用范围 | 关键要求 | 对DS的影响 |
|------|---------|---------|-----------|
| COPPA | 美国 U13 | 家长同意才能收集数据 | U13不能用行为数据训练广告模型 |
| AADC | 英国 U18 | 默认最高隐私保护 | 控制组设计需要Legal预审 |
| DSA | 欧盟 | 推荐系统需可解释，需外部审计 | 模型需要有解释性输出 |

### 实验设计中的合规检查清单

```python
compliance_checklist = {
    'control_group': {
        '问题': '控制组（无干预）在 AADC 下是否合规？',
        '检查': 'AADC 要求默认最高保护 → 完全无保护的控制组可能违规',
        '解决': '控制组改为"软保护"，实验组为"强保护"',
    },
    'data_collection': {
        '问题': 'U13 用户的行为数据能用于实验分析吗？',
        '检查': 'COPPA 要求家长同意才能收集和使用 U13 数据',
        '解决': 'U13 单独建立合规数据管道，或从实验中排除',
    },
    'parental_consent': {
        '问题': '新功能是否需要家长重新同意？',
        '检查': '屏幕时间限制：通知即可 | 数据使用范围扩大：需要重新同意',
        '解决': '在实验启动前，让 Legal 确认同意范围',
    },
    'audit_trail': {
        '问题': 'DSA 要求推荐系统可解释',
        '检查': '实验中的算法变化是否有完整文档和审计日志？',
        '解决': '每次实验记录：变更内容、影响范围、安全评估',
    },
}
```

### 面试金句

> "在每个实验设计里，我会主动过一遍合规检查清单，而不是等Legal来找我。特别是三个关键节点：控制组设计（AADC默认保护要求）、U13数据使用（COPPA家长同意）、算法变更文档（DSA审计要求）。提前解决比事后补救代价小得多——欧盟曾因算法推荐问题对平台开出最高4%全球营收的罚款。"

---

## 12. 面试高频场景题库

### 场景1：举报率异常上升

**标准分析框架：**
```
Step 1: 排除假信号（举报功能改版？分母变化？）
Step 2: 被举报内容的分数分布
  → 低分集中（0-30）: 模型漏检，重训练
  → 高分但未拦截: 阈值太高，降低 + 扩人力
  → 中分进了人审但放过: 标注质量问题，培训审核员
  → 均匀分布: 新型有害内容涌入，更新训练数据
Step 3: 计算恢复到目标Recall需要的人力成本
Step 4: 给CEO汇报（短期+长期方案）
```

### 场景2：新功能 Pre-feasibility 评估

**四个维度必须覆盖：**
```
技术可行性: 指标如何定义和计算？工程实现有无障碍？
样本量可行性: 触发用户比例 × DAU → 功效分析 → 实验周期
法律可行性: 控制组设计是否合规？需要Legal预审
伦理可行性: 控制组是否有基础保护？两组对比程度，不对比有无
```

### 场景3：DAU 下降分析

**拆解三个来源再做诊断：**
```python
dau_decomposition = {
    '新用户获取下降': {
        '原因': '陌生人无法发现账号，病毒传播减弱',
        '判断': '看新用户注册量变化',
        '应对': '政策必然结果，接受',
    },
    '老用户流失增加': {
        '原因': '用户抵制功能，主动离开',
        '判断': '看 churn rate 变化',
        '应对': '优化UI/沟通策略',
    },
    '使用频次下降': {
        '原因': '内容曝光减少，互动动机降低',
        '判断': '看人均 session 数变化',
        '应对': '优化好友内容推荐算法',
    },
}
```

### 场景4：模型是否需要重训练

**三个触发信号：**
```python
retrain_triggers = {
    'auc_degradation': {
        '指标': '红队种子集上的AUC',
        '阈值': 'AUC 下降 > 0.03（相对基线）',
        '周期': '每周监控',
    },
    'appeal_overturn': {
        '指标': '申诉推翻率',
        '阈值': '> 15%（说明模型在某类内容上系统性出错）',
        '周期': '每周监控',
    },
    'false_negative': {
        '指标': '千人标样漏审率',
        '阈值': '> 5%（连续2周）',
        '周期': '每周监控',
    },
}
```

### 场景5：A/B 实验结果解读

**显著但不一定有意义 vs 不显著但可能有意义：**
```python
# 统计显著性 ≠ 实际意义
effect     = 0.001   # D30 retention 提升 0.1个百分点
p_value    = 0.02    # 统计显著

# 但是：
# 0.1% 的 retention 提升对业务有意义吗？
# 这就是为什么 MDE 要在实验前和PM对齐

# 实际意义检验：
mde_agreed = 0.02    # 事前和PM约定的最小有意义效应
if abs(effect) < mde_agreed:
    print("统计显著但无实际意义，不建议全量上线")
    print("建议：观察更长时间，或者接受这个结论并存档")

# 置信区间比 p-value 更有信息量
from scipy import stats
ci = stats.t.interval(0.95, df=n-1, loc=effect, scale=se)
print(f"95%置信区间: [{ci[0]:.4f}, {ci[1]:.4f}]")
# 如果置信区间完全在 MDE 以下 → 可以自信地说"没有实际意义"
```

---

## 速查金句表

| 场景 | 金句 |
|------|------|
| 定阈值 | "阈值是三个约束下的动态平衡：产能给下限，Recall给上限，Precision保效率" |
| AUC | "随机取一个有害和一个无害，模型排对的概率" |
| Ground Truth | "这是个覆盖偏差问题——我只能评估模型发现的，模型漏掉的永远看不到" |
| CUPED | "用实验前行为控制个体差异，方差缩减49%，实验周期减半" |
| RD设计 | "年龄断点是这个领域最干净的自然实验，用户无法操控自己的生日" |
| DiD | "关键是平行趋势假设，上来先画8-12周的预处理趋势图" |
| DAU下降 | "区分政策必然（接受）和设计导致（修复），用安全指标证明是好事" |
| CEO汇报 | "先结论，再证据，最后说需要你拍板什么——2分钟讲完，剩下留给追问" |
| 合规 | "每个实验设计我会主动过合规清单，不等Legal来找我" |
| Family Pairing | "以family_id为随机化单元，同一家庭所有账号分到同一组" |

---

*祝面试顺利！*

---

## 13. 内容审核人审工作流（Feed Pool / CSAT / CSAM）

### 知识点

人审工作流是 Risk Platform 的核心运营组件，DS 的职责是：
设计抽样策略、监控审核质量、建立模型反馈闭环、确保法律合规。

### 完整流水线

```
平台每天上传内容（视频 + 图片）
          ↓
多模态分类模型打风险分数（0-100）
          ↓
三层分流：
  高分（70+）→ 自动拦截
  中分（30-70）→ 人审队列
  低分（0-30）→ 自动放行 + 定期抽样审核
          ↓
Feed Pool 分层抽样
每个国家 1000 条/周
          ↓
CSAT 人工审核
（Content Safety Audit Task）
          ↓
分类处置：
  CSAM  → 强制上报 NCMEC + 立即封号
  CGVR  → 删除 + 账号调查 + 升级专项团队
  普通违规 → 删除 / 限流 / 警告
  误报  → 恢复内容
          ↓
审核结果 → 反馈模型训练数据
```

---

### Feed Pool 1000 Sample 的设计逻辑

**为什么是分层抽样，不是简单随机：**

```python
def design_feed_pool_sample(
    country: str,
    total_sample: int = 1000
) -> dict:
    """
    每个国家每周 1000 条的分层抽样设计
    """
    allocation = {
        # 按分数区间分层（重点放在不确定的中间地带）
        'score_0_30':   int(total_sample * 0.15),  # 150条：检测漏审
        'score_30_50':  int(total_sample * 0.25),  # 250条：模型不确定区
        'score_50_70':  int(total_sample * 0.35),  # 350条：高风险边界
        'score_70_100': int(total_sample * 0.25),  # 250条：验证自动拦截

        # 内容类型比例（按平台上传比例）
        'video': 0.70,
        'photo': 0.30,

        # 高风险类别超采样（确保不被低基率稀释）
        'oversample_categories': [
            'swimwear',       # 边界内容
            'fitness',        # 可能含未成年人
            'beauty_filter',  # 年龄伪装
            'duet_with_minor',# 与未成年人合拍
        ]
    }
    return allocation

# 每个国家单独抽样的原因：
reasons = {
    '文化差异': '不同文化对"有害"的判断标准不同',
    '语言本地化': '本地审核员才能识别本地语言的隐晦表达',
    '监管差异': '欧盟 vs 美国 vs 东南亚的合规要求不同',
    '趋势发现': '某地区新型骚扰手法只在该国样本中出现',
}
```

---

### CSAT 审核员工作内容

**审核员界面包含的信息：**

```
1. 内容本身（视频/图片）
2. 账号信息：
   - 声称年龄 vs 推断年龄
   - 账号注册时间
   - 历史违规记录
   - 关注者中未成年人比例
3. 评论区（可能包含有害互动信号）
4. 模型风险分数和类别分布
5. Policy guideline 快速参考

审核员需要判断：
  ① 内容是否违规（类别）
  ② 严重程度：删除 / 限流 / 警告 / 无操作
  ③ 账号是否需要进一步调查
  ④ 是否需要上报（CSAM 必须上报）
```

**DS 对审核质量的监控：**

```python
import numpy as np
from sklearn.metrics import cohen_kappa_score

# 1. 审核员一致率（IAA）监控
def monitor_iaa(reviewer_a_labels, reviewer_b_labels):
    """
    每周计算审核员之间的一致率
    Kappa > 0.8: 强一致，标注质量好
    Kappa 0.6-0.8: 可接受
    Kappa < 0.6: 需要重新培训或修订标注规范
    """
    kappa = cohen_kappa_score(reviewer_a_labels, reviewer_b_labels)
    print(f"Cohen's Kappa: {kappa:.3f}")
    if kappa < 0.6:
        print("警告：一致率过低，触发标注规范审查")
    return kappa

# 2. 审核员漂移检测
def detect_reviewer_drift(reviewer_id, weekly_stats):
    """
    同一审核员随时间的判断是否发生系统性变化
    比如：审核员疲劳导致越来越宽松
    """
    weekly_violation_rate = [
        w['violation_found'] / w['total_reviewed']
        for w in weekly_stats
    ]

    # 线性趋势检测
    from scipy.stats import linregress
    weeks = list(range(len(weekly_violation_rate)))
    slope, _, _, p_value, _ = linregress(weeks, weekly_violation_rate)

    if p_value < 0.05 and abs(slope) > 0.01:
        direction = "宽松" if slope < 0 else "严格"
        print(f"审核员 {reviewer_id} 判断标准漂移：越来越{direction}")
        print(f"建议：安排校准培训")

# 3. 新审核员校准期
# 前2周：每条审核结果由资深审核员复核
# 通过校准测试（Kappa > 0.75）才独立上岗
```

---

### CSAM 强制上报流程

**CSAM = Child Sexual Abuse Material（儿童性剥削内容）**

```
法律依据（美国）：
  18 U.S.C. § 2258A
  → 平台发现 CSAM 必须在 24-72 小时内上报 NCMEC
  → 违规最高罚款 $150,000/次
  → 平台高管可能承担刑事责任

上报机构：
  美国：NCMEC（National Center for Missing & Exploited Children）
  英国：IWF（Internet Watch Foundation）
  欧盟：各成员国执法机构 + Europol
  全球：INHOPE 网络
```

**DS 维护的技术流程：**

```python
class CSAMDetectionPipeline:
    """
    CSAM 检测使用 hash 匹配，不是 ML 分类
    原因：
    1. 法律原因：平台员工不能直接接触 CSAM 内容
    2. 技术原因：已知 CSAM 的 hash 数据库由 NCMEC 维护
    """

    def __init__(self):
        # PhotoDNA：微软开发的图片 hash 算法
        # 对相似图片（缩放/裁剪/调色）产生相似 hash
        self.photodna_db = self.load_ncmec_hash_database()

        # 视频 hash：帧级别匹配
        self.video_hash_db = self.load_video_hash_database()

    def check_content(self, content_id, content_type, content_bytes):
        """
        每条上传内容都过 hash 检查
        延迟要求：< 100ms（在内容发布前完成检测）
        """
        if content_type == 'image':
            content_hash = self.compute_photodna(content_bytes)
            is_match = content_hash in self.photodna_db
        else:  # video
            frame_hashes = self.compute_video_hashes(content_bytes)
            is_match = any(h in self.video_hash_db for h in frame_hashes)

        if is_match:
            self.trigger_csam_response(content_id)

    def trigger_csam_response(self, content_id):
        """
        发现 CSAM 后的自动响应流程
        """
        # Step 1: 立即从平台移除（不等人工审核）
        self.remove_content_immediately(content_id)

        # Step 2: 账号立即封禁
        account_id = self.get_account(content_id)
        self.suspend_account(account_id)

        # Step 3: 24小时内上报 NCMEC
        report = self.prepare_ncmec_report(
            content_id=content_id,
            account_info=self.get_account_info(account_id),
            ip_address=self.get_upload_ip(content_id),
            upload_timestamp=self.get_upload_time(content_id),
            content_hash=self.get_hash(content_id),
            # 注意：不直接传输内容本身
        )
        self.submit_to_ncmec_cybertip(report)

        # Step 4: 记录审计日志（DSA 合规要求）
        self.log_audit_trail(content_id, 'CSAM_REPORTED')

    def monitor_reporting_metrics(self):
        """
        DS 持续监控的指标
        """
        metrics = {
            'daily_csam_detections': ...,      # 每日检测量
            'reporting_latency_p99': ...,      # 上报延迟P99（法律要求<24h）
            'hash_db_coverage': ...,           # hash 数据库覆盖率
            'false_positive_rate': ...,        # 误报率（需要人工复核）
            'new_hash_additions_weekly': ...,  # 每周新增 hash 数量
        }
        return metrics
```

**关键指标监控：**

```python
# CSAM 上报是唯一不能有 recall/precision 权衡的场景
# "漏报率"没有可接受的阈值 → 目标是零漏报

csam_monitoring = {
    'detection_rate': {
        '定义': '已知 CSAM hash 的检测比例',
        '目标': '> 99.9%',
        '测试': '每周用测试 hash 集验证流水线',
    },
    'reporting_latency': {
        '定义': '从检测到上报 NCMEC 的时间',
        '目标': '< 6 小时（法律要求24小时，内部更严格）',
        '监控': 'P50 / P95 / P99 分位数',
    },
    'false_positive_rate': {
        '定义': '被 hash 匹配但人工复核确认为非 CSAM 的比例',
        '目标': '< 0.1%（hash 算法很准确，误报很少）',
        '原因': '误报 = 正常内容被删除 + 无辜用户被封号',
    },
}
```

---

### CGVR / CGUR 定义

**CGVR = Child Grooming / Vulnerable Risk（诱骗/脆弱风险）**

```
定义：针对未成年人的诱骗行为，不一定是明显违规内容，
      但是有害行为的前兆。

典型行为：
  - 成年人通过私信与未成年人建立不当亲密关系
  - 引导未成年人离开平台（给手机号/约见面）
  - 索要未成年人个人信息（学校/住址/独处时间）
  - 发送边界内容逐步试探
  - 在评论区对未成年人过度赞美外貌

为什么难检测：
  - 单条内容可能无害，需要看行为序列
  - 需要跨会话分析（一个对话里逐步升级）
  - 文化和语境差异大，规则难以泛化

DS 的解决方案：
  不用单条内容分类，而是：
  - 账号行为序列模型（LSTM/Transformer）
  - 检测"与未成年人互动模式"异常的成年账号
  - 图谱分析：一个成年账号关注了多少未成年人？
  - 私信内容分析：出现约见/索取信息等关键词
```

```python
# CGVR 检测：账号行为序列分析
class CGVRDetector:
    def __init__(self):
        # 序列模型：检测成年账号对未成年人的异常行为模式
        self.sequence_model = load_pretrained_transformer()

    def extract_account_features(self, account_id, lookback_days=30):
        features = {
            # 互动对象特征
            'pct_interactions_with_minors': ...,   # 与未成年人互动占比
            'unique_minors_contacted': ...,         # 接触的不同未成年人数
            'avg_minor_account_age_days': ...,      # 接触账号的新旧程度

            # 内容特征
            'compliment_frequency': ...,            # 赞美外貌的频率
            'personal_info_requests': ...,          # 询问个人信息次数
            'offplatform_contact_attempts': ...,    # 尝试转移到平台外的次数

            # 时间特征
            'interaction_time_pattern': ...,        # 是否在学生放学时间活跃
            'escalation_rate': ...,                 # 互动强度是否在升级
        }
        return features

    def score_account(self, account_id):
        features = self.extract_account_features(account_id)
        cgvr_risk_score = self.sequence_model.predict(features)
        return cgvr_risk_score

# 处置流程
def handle_cgvr(account_id, risk_score):
    if risk_score > 0.9:
        # 高置信度 → 账号调查 + 内容审查 + 可能封禁
        escalate_to_trust_safety_team(account_id)
        restrict_minor_interactions(account_id)
    elif risk_score > 0.6:
        # 中等置信度 → 人工审核队列
        add_to_human_review_queue(account_id, priority='high')
    else:
        # 低置信度 → 持续监控
        flag_for_monitoring(account_id)
```

**CGUR（推测定义）：**

```
最可能的解释：
  Child Generated Unsafe Report
  = 由未成年人账号主动发起的举报

或：
  Content Governance Urgent Review
  = 内容治理紧急审核（高优先级队列）

建议面试前确认：
  直接问面试官："我们内部对 CGUR 的定义是什么？"
  展示你知道这是内部术语，不假装知道
  → 这本身就是 Senior level 的表现
```

---

### 各类内容处置矩阵

| 内容类型 | 处置动作 | 时限 | 强制上报？ | 上报机构 |
|---------|---------|------|----------|---------|
| CSAM | 立即删除 + 永久封号 | < 24h | 是（法律） | NCMEC/IWF |
| CGVR 高置信 | 删除 + 账号调查 + 限制 | < 4h | 视情况 | 执法机构 |
| CGVR 中置信 | 人工审核队列 | < 24h | 否 | — |
| 自残/自杀 | 删除 + 危机干预资源 | < 2h | 否（内部升级） | — |
| 暴力内容 | 删除或限流 | < 4h | 否 | — |
| 边界内容 | 限制青少年可见 | < 24h | 否 | — |
| 误报 | 恢复内容 + 道歉通知 | < 48h | 否 | — |

---

### DS 在人审流水线的核心工作

```python
# 1. 覆盖率估计：1000条/国家/周够不够？

def estimate_coverage(sample_size, daily_volume, harmful_rate):
    """
    用1000条样本估计整体有害率的置信区间
    """
    from statsmodels.stats.proportion import proportion_confint

    estimated_harmful = int(sample_size * harmful_rate)
    ci_low, ci_high = proportion_confint(
        estimated_harmful, sample_size, method='wilson'
    )

    print(f"样本量:     {sample_size}")
    print(f"估计有害率: {harmful_rate:.1%}")
    print(f"95% CI:     [{ci_low:.1%}, {ci_high:.1%}]")
    print(f"CI 宽度:    {(ci_high-ci_low):.1%}")

    # 如果 CI 太宽（比如 ±5%），需要增加样本量
    if (ci_high - ci_low) > 0.05:
        print("警告：置信区间过宽，建议增加样本量到2000条")

estimate_coverage(1000, 500000, 0.03)
# 输出：95% CI: [2.1%, 4.2%]，CI宽度 2.1%
# → 1000条对3%有害率的估计是够的

# 2. 模型反馈闭环
def create_training_feedback(human_review_results):
    """
    将人审结果转化为模型训练数据
    关键：处理标注不一致的情况
    """
    # 只使用高一致性的样本
    high_confidence_samples = [
        r for r in human_review_results
        if r['reviewer_agreement_rate'] >= 0.8  # ≥80%审核员一致
    ]

    # 软标签（比硬标签更准确）
    for sample in high_confidence_samples:
        sample['soft_label'] = (
            sample['violation_count'] / sample['total_reviewers']
        )

    return high_confidence_samples

# 3. 上报延迟监控（法律合规）
def monitor_reporting_sla():
    """
    确保 CSAM 上报在法律要求时限内完成
    """
    reports = get_recent_csam_reports(days=7)

    latencies = [
        (r['reported_at'] - r['detected_at']).total_seconds() / 3600
        for r in reports
    ]

    p50  = np.percentile(latencies, 50)
    p95  = np.percentile(latencies, 95)
    p99  = np.percentile(latencies, 99)
    over_24h = sum(1 for l in latencies if l > 24)

    print(f"上报延迟 P50:  {p50:.1f} 小时")
    print(f"上报延迟 P95:  {p95:.1f} 小时")
    print(f"上报延迟 P99:  {p99:.1f} 小时")
    print(f"超过24小时:    {over_24h} 条 ← 法律红线")

    if over_24h > 0:
        alert_legal_and_engineering_team()
```

---

### 面试金句

> "Feed Pool 的1000条抽样不是简单随机，是分层抽样——按内容类型、分数区间、高风险类别超采样，确保人审资源集中在最不确定的地方。每个国家单独抽样的原因是本地化：文化背景、语言、监管要求都不同，一个全球统一的样本会掩盖地区性问题。"

> "CSAM 是整个流水线里唯一有法律强制上报义务的类别——美国法律要求24小时内上报 NCMEC。DS 的工作是确保 PhotoDNA hash 匹配的延迟可控、上报流程自动化、漏报率接近零。这不是一个可以用 recall/precision 权衡的场景——CSAM 的漏报没有可以接受的阈值。"

> "CGVR 和普通内容违规的关键区别是：单条内容可能无害，但行为模式有害。一个成年账号给10个不同的未成年人发送私信、询问他们的学校和放学时间——每条消息单独看可能都不违规，但序列模式是明确的诱骗行为。所以 CGVR 检测用的是账号行为序列模型，不是单条内容分类器。"

> "对于 CGUR 这个缩写，我会直接问面试官内部定义，因为不同团队可能有不同叫法。展示自己知道这是内部术语、不假装知道，本身就是 Senior DS 应有的判断力。"


---

## 14. 合规产品成功指标体系

### 核心思维框架

```
普通产品目标：
  最大化用户价值 → DAU、retention、engagement

合规产品目标：
  在不破坏用户价值的前提下，
  满足监管要求 + 保护用户安全
  → 约束优化问题，不是单一最大化问题
```

### 优先级总览

```
P0（法律红线，不可妥协）：
  CSAM 上报合规率        = 100%
  COPPA 数据合规率       = 100%
  监管投诉响应率         = 100%

P1（安全核心）：
  有害内容曝光率         持续下降
  骚扰举报率变化         下降 ≥ 30%
  CGVR Recall            ≥ 90%

P2（产品健康）：
  Opt-out rate           < 20%
  D30 retention          持平或上升
  好友互动率             不下降

P3（系统健康）：
  模型 AUC               稳定，下降>0.03触发重训练
  漏审率（千人标样）      < 3%
  申诉推翻率             < 15%
  人审产能利用率         70-90%

P4（业务价值）：
  合规成本效率           每季度下降
  监管信任指数           改善
  家长满意度 NPS         > 30
```

---

### P0：监管合规指标（硬性红线）

```python
regulatory_metrics = {

    'CSAM上报合规率': {
        '定义': '在法律要求时限内完成上报的比例',
        '目标': '100%（没有可接受的漏报率）',
        '时限': '美国 < 24h，英国 < 72h',
        '测量': '''
            compliance_rate = (
                reports_within_deadline /
                total_csam_detections
            )
            # 任何 < 100% 都触发法律风险
        ''',
    },

    'COPPA数据合规率': {
        '定义': 'U13 用户数据处理符合 COPPA 要求的比例',
        '目标': '100%',
        '内容': [
            '家长同意率（新注册 U13 账号）',
            'U13 数据不用于广告定向',
            'U13 数据删除请求响应 < 45天',
        ],
    },

    'DSA审计通过率': {
        '定义': '外部审计发现的合规问题数量',
        '目标': '0个重大发现（minor finding 可接受）',
        '周期': '欧盟要求每年一次独立审计',
    },

    '监管投诉处理率': {
        '定义': '在规定时限内回复监管机构投诉的比例',
        '目标': '> 99%，通常30天内',
    },
}
```

---

### P1：安全保护指标

```python
safety_effectiveness_metrics = {

    '有害内容曝光率': {
        '定义': '青少年账号有害内容印象数 / 总印象数',
        '目标': '每季度持续下降',
        '分层': '必须按 U13 / 13-15 / 16-17 分别看',
        '计算': '''
            harmful_exposure_rate = (
                harmful_impressions_to_teens /
                total_impressions_to_teens
            )
        ''',
    },

    '骚扰举报率变化': {
        '定义': '上线合规功能后骚扰类举报的变化幅度',
        '目标': '下降 ≥ 30%',
        '陷阱': '''
            举报率下降有两种解释：
            ① 真实骚扰减少（好事）
            ② 用户觉得举报没用放弃了（坏事）
            → 必须同时监控"举报后行动率"才能区分
        ''',
    },

    '陌生人接触率': {
        '定义': '13-15岁用户收到非好友互动的比例',
        '目标': '仅好友政策上线后下降 ≥ 50%',
        '计算': '''
            stranger_contact_rate = (
                interactions_from_non_friends /
                total_interactions
            )
        ''',
    },

    'CGVR检测准确率': {
        'Precision': '被标记为 CGVR 的账号，真正是诱骗行为的比例',
        'Recall':    '所有真实诱骗行为，被系统检测到的比例',
        '权衡':      '青少年安全场景偏重 Recall',
        '目标':      'Recall ≥ 90%，Precision ≥ 30%',
    },
}
```

---

### P2：用户体验指标

```python
user_experience_metrics = {

    'Opt-out率': {
        '定义': '强制开启安全设置后主动关掉的用户比例',
        '目标': '< 20%',
        '诊断': '''
            opt_out_rate = 60% → 产品设计或沟通问题
            opt_out_rate = 10% → 用户接受默认保护
        ''',
    },

    'D30 Retention对比': {
        '定义': '接受默认保护 vs 选择退出的用户长期留存对比',
        '假设': '健康使用 → 减少疲劳 → 长期留存更好',
        '计算': '''
            retention_protected   = d30_retention(opted_in)
            retention_unprotected = d30_retention(opted_out)
            # 如果 protected >= unprotected
            # → 合规产品不损害留存，向CEO汇报的有力证据
        ''',
    },

    '好友互动率': {
        '定义': '仅好友政策下，好友之间互动质量变化',
        '目标': '好友间互动率上升',
        '信号': '如果连好友互动也下降 → 功能设计有问题',
    },

    '家长满意度 CSAT': {
        '定义': '家长对平台安全措施的满意度',
        '方法': '季度调查，NPS 或 1-5 分',
        '目标': 'NPS > 30',
        '重要性': '家长满意度是监管信任的领先指标',
    },
}
```

---

### P3：系统健康指标

```python
system_health_metrics = {

    '模型AUC（红队种子集）': {
        '定义': '在已知有害内容集上的模型排序能力',
        '目标': '每周监控，下降 > 0.03 触发重训练',
    },

    '人审产能利用率': {
        '定义': '实际送审量 / 最大产能',
        '健康范围': '70-90%',
        '计算': '''
            utilization = daily_queue / (reviewers * cases_per_day)
            # > 95% 持续 → 审核员疲劳 → IAA 下降
            # < 60% → 人力浪费，可以提高阈值节省成本
        ''',
    },

    '申诉推翻率': {
        '定义': '用户申诉后被推翻的比例',
        '目标': '< 15%',
        '触发': '> 15% → 模型在该类内容上系统性误判 → 重训练',
        '分层': '按内容类别看，找出误判集中的类别',
    },

    '千人标样漏审率': {
        '定义': '每周从低分段抽1000条人审，有多少是真实有害',
        '目标': '< 3%',
        '触发': '> 5% 连续2周 → 下调阈值 + 评估重训练',
    },

    'Hash匹配延迟': {
        '定义': '内容上传到 CSAM hash 检测完成的时间',
        '目标': '< 100ms（在内容发布前完成）',
        '法律意义': '超过24小时未检测并上报 = 违法',
    },
}
```

---

### P4：业务价值指标

```python
business_impact_metrics = {

    '监管罚款规避': {
        '定义': '因合规措施避免的潜在罚款金额',
        '估算': '''
            # DSA 最高罚款：全球营收 6%
            # COPPA 单次违规：最高 $51,744
            max_fine_dsa = annual_revenue * 0.06
            # 用于向 CFO 证明安全投入的 ROI
        ''',
    },

    '合规成本效率': {
        '定义': '每发现一个违规内容的人审成本',
        '计算': '''
            cost_per_detection = (
                total_review_cost /
                confirmed_violations
            )
            # = 人审成本 / (Precision × 送审量)
            # Precision 越高 → 每次检测越有效 → 成本越低
        ''',
        '目标': '每季度下降（模型改进 → Precision 提升）',
    },

    '家长信任 → 新用户获取': {
        '定义': '家长满意度对青少年新用户注册的影响',
        '假设': '家长推荐是青少年用户最重要的获客渠道之一',
        '测量': '家长 NPS 与 teen 新注册量的相关性分析',
    },

    '监管信任指数': {
        '定义': '监管机构主动调查的频率变化',
        '代理指标': '媒体负面报道中"TikTok安全问题"的比例',
        '目标': '调查频率下降，主动合作机会增加',
    },
}
```

---

### 面试金句

> "合规产品的指标体系和普通产品根本不同——它是一个约束优化问题：在满足监管红线的前提下最大化用户安全和平台价值，不是单纯最大化 engagement。我把指标分五层：P0 是法律硬性要求不可妥协，P1 是安全有效性，P2 是用户体验（确保合规没有破坏产品），P3 是系统健康，P4 是向 CFO 证明 ROI 的业务价值指标。"

> "一个经常被忽视的指标是 opt-out rate——强制开启安全设置后有多少用户主动关掉。这是合规产品设计质量的最直接信号。如果 opt-out rate = 60%，说明不是安全问题，是产品设计或沟通策略的问题，需要分开诊断。"

> "我会特别警惕'举报率下降=骚扰减少'这个陷阱。举报率下降可能是真实骚扰减少（好事），也可能是用户觉得举报没用放弃了（坏事）。需要同时监控'举报后行动率'才能区分这两种情况。"

> "向 CFO 汇报时，我会把合规投入换算成罚款规避：DSA 最高罚款是全球营收的 6%，COPPA 单次违规最高 $51,744。这让安全投入从'成本中心'变成'风险对冲'，更容易获得资源批准。"

