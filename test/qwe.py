import json
f = open('../data/json_for_lm/harmful_behaviors_filtered-aim.json')
data = json.load(f)
print(data.keys())
print(data['harmful_behaviors'][0]['instruction_with_behavior'])
print(data['method_type'])



