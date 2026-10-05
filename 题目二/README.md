# 题目二：Kaggle 泰坦尼克号生存预测

> 本题目是「机器学习练习」仓库的一部分。请**不要直接在本仓库上修改**，请先按 [提交方式](#六提交方式) 一节，在你自己账号下把仓库建好。
>
> 下文所有相对路径（`data/`、`figures/` 等）都相对于本目录 `题目二/`。

---

## 一、任务背景

### 研究问题

1912 年 4 月 15 日，泰坦尼克号在处女航中撞上冰山沉没，2224 名乘客与船员中有 1502 人遇难。这场海难中**并非所有人都有相同的生还机会**——妇女和儿童被优先安排上救生艇，一等舱乘客的生还率也明显高于三等舱。

本题目要求你利用乘客的**登船信息**（客舱等级、性别、年龄、同行家属人数、船票价格、登船港口等），建立一个**二分类模型**，预测每位乘客是否生还（`Survived = 1` 生还，`0` 遇难）。

这是 Kaggle 平台最经典的入门竞赛（Titanic - Machine Learning from Disaster），也是业界公认的**机器学习项目标准流程**训练场：数据探索 → 缺失值处理 → 特征工程 → 模型选择 → 交叉验证评估 → 生成提交文件。

### 与题目一的关系

题目一（酱油风味 Lasso 降维）聚焦**回归**任务与**正则化/降维**；本题目聚焦**分类**任务，重点考察另外四件事：

1. **训练/测试纪律**——所有预处理参数只能在训练数据上拟合，绝不能"偷看"测试集；
2. **缺失值处理**——真实数据总有空洞，怎么填、填什么、要不要填都是建模决策；
3. **特征工程**——从 `Name`、`Ticket` 这类自由文本里榨取信息，是拉开成绩差距的关键；
4. **分类模型评估**——准确率不是唯一指标，混淆矩阵与 ROC-AUC 才能说明模型到底学到了什么。

---

## 二、数据说明

数据位于 `data/` 目录，来自 Kaggle 竞赛页面，**未做任何改动**：

| 文件 | 行数 | 列数 | 说明 |
| --- | --- | --- | --- |
| `data/train.csv` | 891 | 12 | 训练集，含标签列 `Survived` |
| `data/test.csv` | 418 | 11 | 测试集，**不含** `Survived`，是你需要预测的对象 |
| `data/gender_submission.csv` | 418 | 2 | 官方基线提交样例（只按性别预测：女性=1、男性=0） |

### 字段说明

| 列名 | 含义 | 类型 | 缺失情况 |
| --- | --- | --- | --- |
| `PassengerId` | 乘客编号（1–891 为训练集，892–1309 为测试集） | int | 无 |
| `Survived` | **标签**：0 = 遇难，1 = 生还（仅训练集有） | int | 无 |
| `Pclass` | 客舱等级：1 = 一等舱，2 = 二等舱，3 = 三等舱 | int | 无 |
| `Name` | 全名，含称谓（Mr / Mrs / Miss / Master 等） | str | 无 |
| `Sex` | 性别：male / female | str | 无 |
| `Age` | 年龄（岁），小于 1 岁的记为小数 | float | **训练集缺 177 个，测试集缺 86 个** |
| `SibSp` | 同船的兄弟姐妹 / 配偶人数 | int | 无 |
| `Parch` | 同船的父母 / 子女人数 | int | 无 |
| `Ticket` | 船票编号（字母前缀含舱位信息） | str | 无 |
| `Fare` | 船票价格 | float | 训练集无缺失，**测试集缺 1 个** |
| `Cabin` | 客舱号（首字母为甲板层） | str | **训练集缺 687 个，测试集缺 327 个** |
| `Embarked` | 登船港口：C = 瑟堡，Q = 皇后镇，S = 南安普顿 | str | **训练集缺 2 个** |

> **关于 `Name` 和 `Ticket`**：这两列看起来是"无用文本"，实际上信息量很大。`Name` 里的称谓（`Title`）可以区分"已婚女性 / 未婚女性 / 男性 / 男童"，且能帮助推断缺失的年龄；`Ticket` 的字母前缀有时对应舱位与票价档次。**请至少从 `Name` 中提取 `Title` 特征。**

**读取示例**：

```python
import pandas as pd

train = pd.read_csv('data/train.csv', encoding='utf-8')
test  = pd.read_csv('data/test.csv',  encoding='utf-8')

print(train.shape)   # (891, 12)
print(test.shape)    # (418, 11)
print(train['Survived'].value_counts())   # 0: 549, 1: 342
```

### 训练集 / 测试集的关系

- 两个文件的 `PassengerId` **完全不重叠**，是官方划分好的互斥集合。
- 测试集**没有标签**，因此你无法在本地直接算出测试集的准确率。你的模型好不好，**只能靠训练集上的交叉验证来估计**——这也是本题目要重点训练的判断力。
- 提交文件的 `PassengerId` 必须与 `data/test.csv` 的 `PassengerId` **一一对应**（顺序一致最保险），少一个、多一个、编号写错都会被判为格式错误。

---

## 三、任务要求

完成一整套分类建模流程，并绘制以下**四张图**。

### 图 1：探索性数据分析（EDA）

**必须做成 2×2 的四个子图**（`plt.subplots(2, 2, figsize=(12, 9))`）：

- **(a) 生存率 × 性别 × 客舱等级**：分组柱状图。横轴为客舱等级（1/2/3），每个等级下画两根柱子（female / male），纵轴为**生存率**（0–1）。每根柱子上标注该组样本量 `n`。
- **(b) 年龄分布**：把 `Age` 按 `Survived` 分成两组，画**归一化直方图**（`density=True`，`alpha` 半透明）叠加在同一坐标系中，两条分布要能区分。图例标注 `Survived = 0 / 1`。
- **(c) 票价分布**：同 (b)，按 `Survived` 分组的 `Fare` 归一化直方图。由于 `Fare` 严重右偏，**横轴请用对数刻度**（`plt.xscale('log')`）。
- **(d) 家庭规模与生存率**：定义 `FamilySize = SibSp + Parch + 1`（+1 是乘客本人），计算每个 `FamilySize` 取值的生存率，画柱状图或带数据点的折线图，每点标注样本量 `n`。

> **要点**：图 1 的目的是**用数据说话，找出对生还影响最大的因素**。请在图题或提交说明里用几句话总结你观察到的最强信号。

### 图 2：模型比较（交叉验证）

在训练集上用**分层 5 折交叉验证**比较**至少 4 个模型**，其中**必须包含性别基线**：

| 序号 | 模型 | 说明 |
| --- | --- | --- |
| 1 | **性别基线** | 只用 `Sex` 一个特征的逻辑回归（参照系，你必须打败它） |
| 2 | 逻辑回归 | 使用你构造的全部特征，需标准化 |
| 3 | 随机森林 | `RandomForestClassifier` |
| 4 | 梯度提升 | `GradientBoostingClassifier` 或 `HistGradientBoostingClassifier` |

**图形要求**：**1×2 两个子图**并排——
- **(a) 准确率（Accuracy）**
- **(b) ROC-AUC**

两张子图均为**水平条形图**：纵轴为模型名称，横轴为指标均值；**必须画误差棒**，误差棒长度取**折间标准差**（`std`，不要除以 `sqrt(5)`）；每根柱子末端标注**均值数值**（保留 4 位小数）。

> 注：AUC 需要用**预测概率**（`predict_proba`）计算，不能用 `predict` 的硬标签。

### 图 3：最佳模型评估

选出交叉验证表现最好的模型，做 **1×2 两个子图**：

- **(a) 混淆矩阵**：用 `sklearn.model_selection.cross_val_predict` 得到**全部 891 个训练样本的 out-of-fold 预测标签**（这样每个样本的预测都来自没见过它的那一折，没有信息泄漏），画出 2×2 混淆矩阵，**并在格子里同时标注数量与占比**。坐标轴标注 `Predicted` / `Actual`，类别标签用 `Not Survived` / `Survived`。
- **(b) ROC 曲线**：同样用 `cross_val_predict(..., method='predict_proba')` 得到 out-of-fold 概率，在**同一张图**上画两条线：
  - 5 折**各自的** ROC 曲线（细线，半透明，图例给出各折 AUC）；
  - 用全部 out-of-fold 概率算出的**汇总** ROC 曲线（粗线，图例给出汇总 AUC）；
  - 再画一条**对角虚线**表示随机猜测（AUC = 0.5）。

### 图 4：特征重要性

- **形式**：水平条形图
- **内容**：最佳模型的特征重要性。若最佳模型是树模型，用 `feature_importances_`；若是逻辑回归，用**标准化后**的系数绝对值（或系数本身，请在说明里写清用的是哪个）。
- **排序**：按重要性从大到小排序，**只画前 15 个**特征（若总特征数少于 15 则全部画）。
- **标注**：每根柱子末端写出具体数值。

> 如果你的特征里包含 one-hot 编码后的哑变量（如 `Embarked_C`、`Title_Mr`），请**如实画出哑变量本身**，不要合并——但要能说清它们代表什么。

### 附加要求（写在提交的说明里）

1. 报告图 2 中**每个模型**的 5 折 CV 准确率与 ROC-AUC 均值。
2. 说明你最终选择的模型是哪个，**为什么**（不要只说"它最高"，要结合误差棒谈差异是否显著、以及模型的复杂度/可解释性权衡）。
3. 说明你**每一列缺失值**的处理策略：填什么值、依据是什么、以及为什么不用"直接删行"。
4. 列出你**构造的全部新特征**，每个一句话解释它的建模含义。
5. 报告你的 `submission.csv` 中预测生还的比例（生还人数 / 418），与训练集的生还率（38.4%）对比，并解释差异是否合理。

---

## 四、环境与技术要求

```
Python >= 3.10
pandas, numpy, scikit-learn, matplotlib
```

安装依赖：

```bash
pip install -r requirements.txt
```

### 必须遵守的纪律

- **固定随机种子 `random_state=42`**，保证结果可复现。
- 交叉验证统一用：

  ```python
  from sklearn.model_selection import StratifiedKFold
  cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
  ```

- **所有需要"拟合"的预处理步骤（缺失值填充、标准化、编码）都必须放进 `Pipeline` / `ColumnTransformer`**，让它们在交叉验证的每一折内部、**只用该折的训练部分**拟合。
  > 这是本题目最重要的技术要求。如果你先对全体 891 行做了 `SimpleImputer` 或 `StandardScaler`，再送进交叉验证，就已经发生了**信息泄漏**——CV 分数会虚高，但这不代表模型真的更好。用 `Pipeline` 可以自动避免这个问题。
- **绝对禁止**在 `test.csv` 上做任何形式的 `fit`（包括 `fit_transform`、用它计算均值/中位数来填充）。测试集只能被 `transform` / `predict`。
- 图片统一 **300 dpi** 保存。
- 图中文字用英文（图题、坐标轴、图例），版面风格对齐学术论文插图；字体建议 Times New Roman。
- 代码需**可一键运行**，路径使用相对路径。

---

## 五、提交内容

请在你仓库的 `题目二/` 目录下，提交以下内容：

```
题目二/
├── README.md              # 简要说明你的思路、附加要求的 5 项结果
├── titanic_analysis.py    # 你的分析代码（或 .ipynb 笔记本）
├── submission.csv         # 测试集预测结果（Kaggle 格式，418 行 + 表头）
├── figures/
│   ├── fig1_eda.png
│   ├── fig2_model_comparison.png
│   ├── fig3_model_evaluation.png
│   └── fig4_feature_importance.png
└── requirements.txt       # 你实际使用的依赖
```

`题目二/figures/` 目录已在本仓库中留空，你的四张图请输出到这里（文件名可自定，但请保持可辨识）。

### submission.csv 格式

**两列，表头必须完全一致，且不能有多余的行或列：**

```csv
PassengerId,Survived
892,0
893,1
894,0
```

自查：

```python
import pandas as pd

sub  = pd.read_csv('submission.csv')
test = pd.read_csv('data/test.csv')

assert list(sub.columns) == ['PassengerId', 'Survived'], sub.columns
assert len(sub) == 418, len(sub)
assert sub['Survived'].isin([0, 1]).all(), 'Survived 只能取 0 或 1'
assert set(sub['PassengerId']) == set(test['PassengerId']), '编号与 test.csv 不匹配'
assert sub['PassengerId'].is_unique, '有重复编号'
print('格式检查通过；预测生还人数 =', int(sub['Survived'].sum()))
```

> `Survived` 请提交 **0/1 整数标签**，不要提交概率（如 `0.83`），也不要提交字符串。

---

## 六、提交方式

**一、在自己账号下建仓库**

1. 打开「机器学习练习」仓库页面
2. 点击右上角的 **Use this template** → **Create a new repository**
3. 在弹窗中，在**你自己账号**下创建一个**同名仓库**（可见性建议选 **Private**）
   > 你已经是 `YileWang-Lab` 组织的成员，也可以把仓库建在组织下（Owner 选 `YileWang-Lab`）；两种方式都可以，**只要不建在本模板仓库里**。

**二、clone 到本地做作业**

4. 把**你自己账号下**的仓库 clone 到本地：

   ```bash
   git clone https://github.com/<你的用户名>/Machine-Learning-Practice.git
   cd Machine-Learning-Practice/题目二
   ```

5. 在本地完成建模与四张图的绘制，把图片输出到 `figures/`，把提交文件命名为 `submission.csv`

**三、push 回去**

6. 提交并推送回你自己账号下的仓库：

   ```bash
   git add .
   git commit -m "完成题目二"
   git push
   ```

7. 把仓库 URL 交给任课老师；如需老师直接查看，在仓库 **Settings → Collaborators** 中把老师加为协作者

> **请勿直接向本模板仓库 push 代码**，也请勿在本仓库开分支或提 Pull Request。本仓库是所有同学的公共任务书与数据源，作业一律提交在你自己账号下的副本仓库中。

---

## 七、自查（确认你做对了）

跑完后可以和下面的参考值对一下，**如果你算出来的数明显不同，说明中间某一步有问题**：

**数据规模与分布**

- 训练集 **891 行**，其中生还 **342 人**，生还率 **38.4%**
- 训练集中女性生还率 **74.2%**（314 人中 233 人生还），男性生还率 **18.9%**（577 人中 109 人生还）
- 测试集 **418 行**，`PassengerId` 从 **892 到 1309**，与训练集**无重叠**

**缺失值个数（务必核对，填错了会静默影响结果）**

| 列 | 训练集 | 测试集 |
| --- | --- | --- |
| `Age` | 177 | 86 |
| `Cabin` | 687 | 327 |
| `Embarked` | 2 | 0 |
| `Fare` | 0 | **1** |

> 注意测试集也缺 `Fare` 和 `Age`。如果你只在训练集上处理缺失值，测试集会崩在 `predict` 那一步。这也是"必须把 imputer 放进 Pipeline"的实际好处。

**基线参考值**

- 只用 `Sex` 的性别基线，5 折 CV 准确率 **≈ 0.7868**（折间标准差约 0.019，各折约 0.764–0.820）
- `data/gender_submission.csv` 由该规则生成，预测 **152 人**生还（生还率 **36.4%**）
- **你的模型 CV 准确率应当明显高于 0.7868。** 一个不调参的参考实现（特征为 `Title` + `FamilySize` + `IsAlone` + `HasCabin` + 原始列，全部预处理走 `Pipeline`）大致能得到 **准确率 0.81–0.84、AUC 0.87–0.88**；其中逻辑回归与梯度提升约 0.83，随机森林约 0.81。
- 如果只做到 0.78 左右，基本等于没比"只看性别"多学到东西，请回头检查特征工程与缺失值处理；如果做到 0.90 以上，几乎可以肯定是**信息泄漏**（比如把 `Survived` 漏进了特征，或先对全体数据做了填充/标准化再交叉验证）。

> 说明：本节的参考值用于帮你定位错误（例如读错列、交叉验证折数不对、误用了 `predict` 而非 `predict_proba` 算 AUC 等），请仍然独立完成分析过程。

---

## 八、附录

### A. 图 1 的坐标轴标签中英文对照

matplotlib 默认字体不含中文字形，直接画会把标签渲染成方框（俗称"豆腐块"）。学术插图通常用英文标注，请使用下表英文名。若确需中文，可设置：

```python
plt.rcParams['font.sans-serif'] = ['SimHei']   # 或 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False     # 修正负号显示为方块的问题
```

| 列名 | 图上的英文标签 |
| --- | --- |
| `Survived` | Survived / Not Survived |
| `Pclass` | Passenger Class |
| `Sex` | Sex |
| `Age` | Age (years) |
| `SibSp` | # of Siblings / Spouses |
| `Parch` | # of Parents / Children |
| `Fare` | Fare |
| `Cabin` | Cabin |
| `Embarked` | Port of Embarkation |
| 生还率 | Survival Rate |
| 准确率 | Accuracy |
| 特征重要性 | Feature Importance |

### B. 特征工程思路提示

下面是一些常见方向，**不要求全部实现**，但请至少完成 `Title` 与 `FamilySize`：

- **`Title`**：从 `Name` 中用正则提取称谓（`, Mr.` → `Mr`），把低频称谓（`Dr`、`Rev`、`Col`、`Capt`、`Countess` 等）合并为 `Rare`。可进一步区分 `Master`（男童，生还率高）与 `Mr`。
- **`FamilySize`**：`SibSp + Parch + 1`；再派生 `IsAlone`（`FamilySize == 1`）。
- **`HasCabin`**：`Cabin` 是否有值（有舱位记录的多为一等/二等舱，是"社会阶层"的强代理变量）。
- **`Age` 分箱**：把连续年龄切成几段（如 Child / Teen / Adult / Senior）。
- **`Fare` 分箱**或 `log(Fare)`：缓解右偏。
- **`Ticket` 前缀**：取船票编号的字母前缀（无前缀的单独归为一类）。
- **`Embarked`**：one-hot 编码（这是**名义变量**，不要用 0/1/2 编码后当数值用）。

> 注意 `Pclass` 是有序变量（1 > 2 > 3），可以直接当数值用，也可以 one-hot；两种做法都请说明理由。

### C. 数据来源与引用

- 竞赛主页：<https://www.kaggle.com/competitions/titanic>
- 数据版权归 Kaggle 及原始提供方所有。本仓库仅将**原始竞赛数据**（`train.csv` / `test.csv`，均未改动）用于课程教学，请勿外传或用于商业用途。
- `data/gender_submission.csv` 为按官方基线规则（女性预测生还、男性预测遇难）生成，用途是给学生对照提交格式。
