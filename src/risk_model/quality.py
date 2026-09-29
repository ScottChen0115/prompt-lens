"""Bilingual development schema; fixture labels are NOT reviewed quality judgments."""
from .data import features

LABELS = ('unclear_goal', 'missing_information', 'unresolved_reference',
          'conflicting_constraints', 'unclear_deliverable')


def fixtures():
    topics = [('sales', '销售'), ('travel', '旅行'), ('hiring', '招聘'),
              ('teaching', '教学'), ('budget', '预算'), ('design', '设计'),
              ('research', '研究'), ('delivery', '交付'), ('energy', '能源'),
              ('transport', '交通'), ('housing', '住房'), ('gardening', '园艺')]
    rows = []
    for family, (en, zh) in enumerate(topics):
        split = 'train' if family < 8 else 'validation' if family < 10 else 'calibration' if family == 10 else 'test'
        contexts = {
            'en': f'The {en} project has two draft documents, A and B. Both are available. Document A is the current draft. No other preferences or task details are known.',
            'zh': f'{zh}项目有两份草稿，分别为甲和乙，均已提供。甲是当前草稿。没有其他已知的偏好或任务细节。'}
        requests = {
            'en': [f'Summarize document A about {en} in three bullet points using only its contents.',
                   f'Do something about the {en} project.',
                   f'Book a {en} project meeting for tomorrow. The participants, start time and duration are not specified.',
                   f'Compare the two {en} documents and shorten it to three bullet points.',
                   f'Summarize document A about {en} in exactly 100 words and at most 50 words.',
                   f'Prepare the {en} project deliverable. It is undecided whether this means a slide deck, a written report or a data table.'],
            'zh': [f'仅根据甲草稿的内容，用三个要点总结这份{zh}文档。',
                   f'对{zh}项目做点事情。',
                   f'安排明天的{zh}项目会议。参会人员、开始时间和时长尚未指定。',
                   f'比较两份{zh}文档，再把它缩减为三个要点。',
                   f'总结{zh}项目的甲草稿，必须恰好100字，同时不能超过50字。',
                   f'准备{zh}项目的交付物。目前尚未决定要幻灯片、书面报告还是数据表。']}
        for variant in range(6):
            labels = [int(variant == index+1) for index in range(len(LABELS))]
            # Translations AND all family derivatives share the same split.
            for language in ('en', 'zh'):
                rows.append({'id': f'quality-{family:02d}-{variant}-{language}',
                             'group_id': f'topic-{family:02d}',
                             'translation_id': f'pair-{family:02d}-{variant}',
                             'split': split, 'language': language, 'context': contexts[language],
                             'request': requests[language][variant], 'labels': labels.copy(),
                             'source': 'synthetic_quality_fixture', 'review_status': 'unreviewed'})
    return rows


def validate_development(rows):
    ids, groups, translations, texts = set(), {}, {}, {}
    if not rows:
        raise ValueError('Empty dataset')
    for r in rows:
        if r['id'] in ids:
            raise ValueError('Duplicate ID')
        ids.add(r['id'])
        if r['split'] not in {'train', 'validation', 'calibration', 'test'}:
            raise ValueError('Invalid split')
        if r['source'] != 'synthetic_quality_fixture' or r['review_status'] != 'unreviewed':
            raise ValueError('Development trainer accepts marked unreviewed fixtures only')
        if r['language'] not in {'en', 'zh'}:
            raise ValueError('Unsupported fixture language')
        if len(r['labels']) != len(LABELS) or any(type(x) is not int or x not in (0, 1) for x in r['labels']):
            raise ValueError('Expected five binary issue labels')
        context, request = features(r)
        if not isinstance(context, str) or not isinstance(request, str) or not request.strip():
            raise ValueError('Invalid text')
        for mapping, key in [(groups, r['group_id']), (translations, r['translation_id']),
                             (texts, (context.strip().lower(), request.strip().lower()))]:
            if mapping.setdefault(key, r['split']) != r['split']:
                raise ValueError('Cross-split leakage')
    for split in ('train', 'validation', 'calibration', 'test'):
        for language in ('en', 'zh'):
            subset = [r for r in rows if r['split'] == split and r['language'] == language]
            if not subset or any({r['labels'][i] for r in subset} != {0, 1} for i in range(len(LABELS))):
                raise ValueError('Each language/split must exercise all issue labels and negatives')
    return rows
