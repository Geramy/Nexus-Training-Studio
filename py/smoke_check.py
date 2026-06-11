#!/usr/bin/env python3
"""Quick structural check that generated training data is serve-exact:
BOARD STATE injections present, tool results are the executor's prose strings
(not JSON), discovery carries the PROJECT BASELINE, and the recovery/skip
builders are in the mix. Run from the studio root: python3 py/smoke_check.py"""
import ast
import json
import sys

sys.path.insert(0, 'py')
import gen_training_data as g  # noqa: E402

convos = g.generate(60, {'setup', 'discovery', 'tasks'}, 11)
print(f"generated {len(convos)} conversations OK")

su = [c for c in convos if 'BOARD STATE' in json.dumps(c['messages'])]
print('convos with BOARD STATE:', len(su))
bs = [m for m in su[0]['messages']
      if m['role'] == 'system' and 'BOARD STATE' in m['content']]
print('--- last board state in convo 1 ---')
print(bs[-1]['content'][:240])

tr = [m for c in su[:6] for m in c['messages'] if m['role'] == 'tool']
bad = [m for m in tr if m['content'].lstrip().startswith('{')]
print('tool results:', len(tr), '| still-JSON:', len(bad), '|',
      sorted({m.get('name') for m in bad}))
for m in tr[:7]:
    print(' •', m.get('name'), '→', m['content'][:95])

dis = [c for c in convos if 'PROJECT BASELINE' in c['messages'][0]['content']]
print('discovery convos with baseline:', len(dis))
for m in [m for m in dis[0]['messages'] if m['role'] == 'tool'][:5]:
    print(' •', m.get('name'), '→', m['content'][:95])

js = json.dumps(convos)
print('nudge-record:', 'NOT recorded the user' in js,
      '| nudge-continue:', 'Continue: take the next step' in js,
      '| skip:', 'skipped the question' in js)
ast.parse(open('py/eval_interview.py').read())
print('eval_interview.py parses OK')
