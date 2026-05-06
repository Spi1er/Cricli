# 标题候选与最终推荐中文样例

这份样例从 `data/processed/headline_review_demo_cases.csv` 中挑选了 5 篇文章，用于中文汇报或 demo 展示。英文标题保留原实验输出，中文标题是对应语义翻译，不参与模型评测。

## 样例 1：NASA 太空行走

- Seed ID: 1
- Category: news
- Objective: Editorial
- Article summary: International Space Station astronauts face physically demanding spacewalks that can last for hours.

| 角色 | 英文标题 | 中文标题 |
| --- | --- | --- |
| Human baseline | NASA's Christina Koch got a little bit messy during first all-female spacewalk | NASA 的克里斯蒂娜·科赫在首次全女性太空行走中遇到了一些小状况 |
| Low-risk alternative | NASA Astronauts Face Challenges of Long Spacewalks on ISS | NASA 宇航员在国际空间站面对长时间太空行走挑战 |
| Recommended | NASA Astronauts on ISS Face Challenges of Long Spacewalks | 国际空间站 NASA 宇航员面对长时间太空行走挑战 |

最终推荐：**国际空间站 NASA 宇航员面对长时间太空行走挑战**

推荐理由翻译：该标题更偏向平衡、简洁、可发布的新闻标题；标题党风险低；受众/角色评分更偏好它；本地质量 critic 也给出了较高评分。

## 样例 2：西雅图海鹰击败 49 人

- Seed ID: 4
- Category: sports
- Objective: Growth
- Article summary: The Seattle Seahawks rallied to beat the previously unbeaten 49ers, showing the team's competitive mindset.

| 角色 | 英文标题 | 中文标题 |
| --- | --- | --- |
| Human baseline | We always believe': Win over 49ers proves Seattle Seahawks' mindset is more than just lip service | “我们一直相信”：击败 49 人证明西雅图海鹰的心态并非空话 |
| Low-risk alternative | Seattle Seahawks win over 49ers | 西雅图海鹰击败 49 人 |
| Recommended | Seattle Seahawks Defeat Unbeaten 49ers in a Thrilling Comeback Victory | 西雅图海鹰惊险逆转，击败此前不败的 49 人 |

最终推荐：**西雅图海鹰惊险逆转，击败此前不败的 49 人**

推荐理由翻译：该标题在控制标题党和信任风险的同时，更偏向有吸引力的增长目标；标题党风险低；受众/角色评分更偏好它；本地质量 critic 评分较高；主要信息也能被文章摘要支持。

## 样例 3：卫生纸婚纱

- Seed ID: 37
- Category: lifestyle
- Objective: Editorial
- Article summary: Wedding gowns were made from Quilted Northern toilet paper, tape, glue, needle, and thread.

| 角色 | 英文标题 | 中文标题 |
| --- | --- | --- |
| Human baseline | These 12 exquisite wedding dresses are made from toilet paper | 这 12 条精致婚纱竟然是用卫生纸做成的 |
| GenAI baseline | Wedding Gowns Created from Quilted Northern Toilet Paper and Craft Supplies | 用 Quilted Northern 卫生纸和手工材料制成的婚纱 |
| Low-risk alternative | Artistry in Unconventional Materials: Wedding Gowns from Toilet Paper | 非传统材料中的工艺：用卫生纸制成的婚纱 |
| Recommended | Wedding gowns made from quilted Northern toilet paper, tape, glue, and a needle and thread | 用 Quilted Northern 卫生纸、胶带、胶水、针线制成的婚纱 |

最终推荐：**用 Quilted Northern 卫生纸、胶带、胶水、针线制成的婚纱**

推荐理由翻译：该标题更偏向平衡、简洁、可发布的新闻/生活方式标题；本地质量 critic 给出了较高评分；标题中的主要信息能被文章摘要支持。

## 样例 4：Senior Santa 公益项目

- Seed ID: 65
- Category: travel
- Objective: Editorial
- Article summary: Volunteers bring holiday cheer to seniors in long-term care facilities through Brevard County TRIAD's Senior Santa program.

| 角色 | 英文标题 | 中文标题 |
| --- | --- | --- |
| Human baseline | Senior Santa a great way to help area's elderly | “Senior Santa” 是帮助当地老年人的好方式 |
| Low-risk alternative | Annual Senior Santa Program Brightens Holidays for Seniors in Long-Term Care | 年度 Senior Santa 项目为长期护理机构老人点亮节日 |
| Recommended | Volunteers Spread Holiday Cheer to Seniors Through Senior Santa Program in Brevard County | 志愿者通过布里瓦德县 Senior Santa 项目为老人送去节日温暖 |

最终推荐：**志愿者通过布里瓦德县 Senior Santa 项目为老人送去节日温暖**

推荐理由翻译：该标题更偏向平衡、简洁、可发布的标题；标题党风险低；受众/角色评分更偏好它；本地质量 critic 给出了较高评分；主要信息能被文章摘要支持。

## 样例 5：密苏里审计传票

- Seed ID: 99
- Category: news
- Objective: Growth
- Article summary: Missouri State Auditor Nicole Galloway issued another subpoena to Clay County officials for documents related to a citizen-mandated audit.

| 角色 | 英文标题 | 中文标题 |
| --- | --- | --- |
| Human baseline | State auditor again issues subpoena to Clay County in citizen-mandated audit | 州审计员再次就公民授权审计向克莱县发出传票 |
| GenAI baseline | Missouri Auditor Issues Subpoena for Clay County Audit Documents | 密苏里州审计员就克莱县审计文件发出传票 |
| Low-risk alternative | Missouri State Auditor Seeks Documents from Clay County Amid Audit Issues | 密苏里州审计员在审计问题中要求克莱县提交文件 |
| Recommended | Missouri Auditor Nicole Galloway Issues Subpoena to Clay County Officials | 密苏里州审计员妮科尔·加洛韦向克莱县官员发出传票 |

最终推荐：**密苏里州审计员妮科尔·加洛韦向克莱县官员发出传票**

推荐理由翻译：该标题在控制标题党和信任风险的同时，更偏向增长目标；标题党风险低；本地质量 critic 给出了较高评分；主要信息能被文章摘要支持。

## 使用建议

这些中文翻译适合放在报告或演示页中，用来解释系统不是简单地“生成一个标题”，而是在多个候选标题之间做选择：

1. Human baseline 展示原始人工标题。
2. GenAI baseline 展示直接生成式 AI 的强基线。
3. Low-risk alternative 展示系统如何找到更低风险的备选。
4. Recommended 展示最终根据目标函数、critic 和 persona 选择出的标题。

中文翻译只用于展示，不应该替代英文标题进入当前模型评测流程。
