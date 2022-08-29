import json
from sklearn.model_selection import train_test_split

directories = ['cars', 'ceremonies', 'food', 'indoor', 'ashkhas', 'sport']

seed = {'cars': 13, 'ceremonies': 65, 'food': 2, 'indoor': 2, 'ashkhas':25, 'sport': 102}
train_size = 16
for d in directories:
    with open(f'{d}.json', 'r', encoding='utf-8') as f:
        dataset = json.load(f)
        train, test = train_test_split(dataset, test_size=len(dataset)-train_size, random_state=seed[d])
    
        with open(f'{d}_train.json', 'w', encoding='utf-8') as f:
            f.write(json.dumps(train, ensure_ascii=False))
        with open(f'{d}_test.json', 'w', encoding='utf-8') as f:
            f.write(json.dumps(test, ensure_ascii=False))