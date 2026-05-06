# Build a Chinese-localized copy of the static headline review console.

from __future__ import annotations

import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


DEFAULT_INPUT = PROJECT_ROOT / "demo" / "headline_review_console.html"
DEFAULT_OUTPUT = PROJECT_ROOT / "demo" / "headline_review_console_zh.html"

TRANSLATIONS = [
    ("Headline Review Console", "标题审核控制台"),
    ("Headline recommendation", "标题推荐"),
    ("Local demo", "本地演示"),
    ("Search articles or headlines", "搜索文章或标题"),
    ("articles", "篇文章"),
    ("objectives", "个目标"),
    ("visible rows", "条展示候选"),
    ("No matching articles.", "没有匹配的文章。"),
    ("Article", "文章"),
    (" article #", " 文章 #"),
    ("Internal candidates", "内部候选数"),
    ("Generator sources", "候选来源数"),
    ("Recommended headline", "推荐标题"),
    ("Visible decision set", "可见决策选项"),
    ("Risk safety", "风险安全"),
    ("Quality", "质量"),
    ("Audience fit", "受众匹配"),
    ("Objective fit", "目标适配"),
    ("Support", "证据支持"),
    ("Selected", "已选中"),
    ("Candidate", "候选标题"),
    ("Untitled", "无标题"),
    ("Decision ", "决策分 "),
    ("Final ", "综合分 "),
    (" recommendation", " 推荐"),
    ("Trust / Safety", "信任 / 安全"),
    ("Trust/Safety", "信任 / 安全"),
    ("Safety", "安全"),
    ("Growth", "增长"),
    ("Editorial", "编辑平衡"),
    ("Specificity", "具体性"),
    ("Human baseline", "人工基线"),
    ("GenAI baseline", "GenAI 基线"),
    ("Low-risk alternative", "低风险备选"),
    ("Recommended", "推荐结果"),
    ("Original human-written headline used as the editorial reference point.", "原始人工标题，作为编辑参考基线。"),
    ("Selected from the hidden candidate pool for low clickbait risk while keeping predicted quality.", "从隐藏候选池中选出的低标题党风险备选，同时尽量保持预测质量。"),
    ("Prioritizes faithful, clear, low-risk headlines for sensitive publishing contexts.", "优先选择忠实、清晰、低风险的标题，适合更敏感的发布场景。"),
    ("Prioritizes engaging headlines while keeping clickbait risk under control.", "优先选择更有吸引力的标题，同时控制标题党风险。"),
    ("Prioritizes balanced, compact, publication-ready news headlines.", "优先选择平衡、简洁、可发布的新闻标题。"),
    ("Prioritizes concrete details and strong support from the article summary.", "优先选择具体细节更多、且更能被文章摘要支持的标题。"),
    ("It has low clickbait risk.", "它的标题党风险较低。"),
    ("It is favored by persona/audience scoring.", "它在受众/角色偏好评分中表现较好。"),
    ("The local quality critic rates it highly.", "本地质量 critic 给出了较高评分。"),
    ("It has strong support from the article summary.", "它与文章摘要的证据支持较强。"),
    ("It is a stronger fit for the selected operating objective.", "它更符合当前选择的业务目标。"),
    ("Selected by the persona-calibrated Trust / Safety objective.", "由受众校准后的信任/安全目标选中。"),
    ("Selected by the persona-calibrated Growth objective.", "由受众校准后的增长目标选中。"),
    ("Selected by the persona-calibrated Editorial objective.", "由受众校准后的编辑平衡目标选中。"),
    ("Selected by the persona-calibrated Specificity objective.", "由受众校准后的具体性目标选中。"),
    ("Prioritizes factual, clear, non-clickbait headlines for trust-sensitive surfaces.", "优先选择事实准确、清晰、非标题党的标题，适合重视信任的展示场景。"),
    ("Prioritizes engaging headlines while still controlling clickbait and trust risk.", "优先选择更有吸引力的标题，同时控制标题党和信任风险。"),
    ("Prioritizes concrete, source-supported details without losing clarity.", "优先选择具体、可由来源支持的细节，同时保持清晰。"),
    ("Risk is low after folding clickbait into the safety score.", "将标题党风险融合进安全分后，该标题风险较低。"),
    ("Audience/persona scoring favors it.", "受众/角色评分更偏好它。"),
    ("Its main terms are supported by the article summary.", "标题中的主要信息能被文章摘要支持。"),
    ("Not selected because it has higher risk by ", "未被选中，因为它的风险更高 "),
    ("Not selected because it has lower quality by ", "未被选中，因为它的预测质量更低 "),
    ("Not selected because it has weaker Trust / Safety objective fit by ", "未被选中，因为它对信任/安全目标的适配更弱 "),
    ("Not selected because it has weaker Growth objective fit by ", "未被选中，因为它对增长目标的适配更弱 "),
    ("Not selected because it has weaker Editorial objective fit by ", "未被选中，因为它对编辑平衡目标的适配更弱 "),
    ("Not selected because it has weaker Specificity objective fit by ", "未被选中，因为它对具体性目标的适配更弱 "),
    ("Not selected because it has weaker audience fit by ", "未被选中，因为它的受众匹配更弱 "),
    ("Not selected because it has less summary support by ", "未被选中，因为它的摘要证据支持更弱 "),
    ("; higher risk by ", "；风险更高 "),
    ("; lower quality by ", "；预测质量更低 "),
    ("; weaker Trust / Safety objective fit by ", "；信任/安全目标适配更弱 "),
    ("; weaker Growth objective fit by ", "；增长目标适配更弱 "),
    ("; weaker Editorial objective fit by ", "；编辑平衡目标适配更弱 "),
    ("; weaker Specificity objective fit by ", "；具体性目标适配更弱 "),
    ("; weaker audience fit by ", "；受众匹配更弱 "),
    ("; less summary support by ", "；摘要证据支持更弱 "),
    ("Kept as the editorial reference; the recommendation has a stronger combined decision score.", "保留为编辑参考基线；推荐标题拥有更强的综合决策分。"),
    ("Strong baseline candidate, but the selector found a better match for this objective.", "这是一个强基线候选，但选择器找到了更符合当前目标的标题。"),
    ("Safe alternative, but the recommendation gives a better overall tradeoff.", "这是一个安全备选，但推荐标题给出了更好的整体权衡。"),
    ("Close alternative, but not the best combined tradeoff for this objective.", "这是一个接近的备选，但不是当前目标下的最佳综合权衡。"),
]


def localize_html(source: str) -> str:
    localized = source
    for old, new in TRANSLATIONS:
        localized = localized.replace(old, new)
    localized = localized.replace('<html lang="en">', '<html lang="zh-CN">')
    localized = localized.replace('<title>标题审核控制台</title>', '<title>标题审核控制台</title>')
    return localized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Chinese-localized copy of the headline review console.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.input.read_text(encoding="utf-8")
    localized = localize_html(source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(localized, encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
