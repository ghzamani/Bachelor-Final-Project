import json
from itertools import groupby
from operator import itemgetter
from parsivar import Normalizer

def preprocess_caption(caption):
    my_normalizer = Normalizer(statistical_space_correction=True)
    caption = my_normalizer.normalize(caption)
    if caption[-1] not in ['.', '!', '?', '.', '؟', '!']:
        return caption + '.'

with open('captions_newest.json', 'r', encoding='utf-8-sig') as f:
    dataset = json.load(f)

categories = {}
directories = ['cars', 'ceremonies', 'food', 'indoor', 'ashkhas', 'sport']
for d in directories:
    categories[d] = []

for d in dataset:
    cat = d['image'].split('/')[1]
    d['caption'] = preprocess_caption(d['caption'])
    categories[cat].append(d)

dataset = {}
for d in directories:
    grouped_captions = []
    sort = sorted(categories[d], key=itemgetter('image_id'))
    iter = groupby(sort, key=itemgetter('image_id'))
    for key, group in iter:
        caps = list(group)
        grouped_captions.append({"id": key, "caption": [i['caption'] for i in caps], "captions_ids": [i['id'] for i in caps], "image": caps[0]['image']})

    dataset[d] = grouped_captions

for cat in dataset.keys():
    with open(f'{cat}.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(dataset[cat], ensure_ascii=False))