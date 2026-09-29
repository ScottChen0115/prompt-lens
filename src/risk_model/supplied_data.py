"""Read supplied datasets as data only; never import or execute their generators."""
from .quality import LABELS

BRQ_LABELS = ('unclear_goal','missing_essential_information','ambiguous_reference',
              'conflicting_requirements','unclear_deliverable')


def render_context(value):
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        raise ValueError('Context must be text or a list of declared context items')
    lines=[]
    for item in value:
        kind=item['type']
        if kind=='message':
            lines.append(f"{item['role']}: {item['text']}")
        elif kind=='attachment':
            if item.get('available') and isinstance(item.get('text'),str):
                lines.append(f"[attachment: {item['name']}]\n{item['text']}")
            else:
                # Never serialize hidden attachment contents, including a literal 'None'.
                lines.append(f"[attachment: {item['name']} — content not available]")
        elif kind=='omitted_history':
            lines.append(f"[{item['messages']} earlier messages not available]")
        elif kind=='profile':
            lines.append(f"[saved user profile] {item['text']}")
        else:
            raise ValueError(f'Unknown context item: {kind}')
    return '\n'.join(lines)


def normalize(row, source, split):
    if row.get('split') != split:
        raise ValueError('Source split mismatch')
    if not isinstance(row.get('request'),str) or not row['request'].strip():
        raise ValueError('Nonempty request required for development rows')
    if row.get('language') not in ('en','zh'):
        raise ValueError('Unexpected development language')
    names=BRQ_LABELS if source=='brq' else LABELS
    values=[]
    for name in names:
        value=row['labels'][name]
        if isinstance(value,dict): value=value['value']
        if value is None or value=='unknown': values.append(None)
        elif type(value) is int and value in (0,1): values.append(value)
        elif value in ('0','1'): values.append(int(value))
        else: raise ValueError(f'Invalid label value: {value!r}')
    relations={}
    for key in ('group_id','pair_id','paraphrase_set','conversation_id','translation_family_id','scenario_id'):
        if row.get(key): relations[key]=str(row[key])
    return {'id':source+':'+row['id'],'source_id':row['id'],'source':source,'split':split,
            'language':row['language'],'context':render_context(row['context']),
            'request':row['request'],'labels':values,'relations':relations,
            'review_status':'machine_labels_not_human_verified'}
