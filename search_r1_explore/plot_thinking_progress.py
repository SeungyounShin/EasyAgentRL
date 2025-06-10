import os 
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

PATH = '/home/robin/verl/validation_data_log/qwen2.5-7b-instruct'

json_files = [os.path.join(PATH, f) for f in os.listdir(PATH) if f.endswith('.jsonl')]
# sort by -> 100, 200, ....
json_files.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
json_files = json_files[:7]

step = [0, 10, 20, 30, 40, 50, 60]
thinking_presence = []
accuracy = []

for json_file in tqdm(json_files, desc='Processing JSON files'):
    _thinking_presence = []
    _accuracy = []
    with open(json_file, 'r') as f:
        for line in tqdm(f, desc='Processing lines'):
            data = json.loads(line)
            
            prompt : str = data['input']
            question : str = prompt.split('Question:')[1].split('assistant')[0].strip()
            response : str = data['output']
            score : int = data['reward']

            # count <think>
            think_count = response.count('<think>')
            _thinking_presence.append(think_count)
            _accuracy.append(score)

    thinking_presence.append(sum(_thinking_presence) / len(_thinking_presence))
    accuracy.append(sum(_accuracy) / len(_accuracy))

# plot
fig, ax1 = plt.subplots()

# ――― 왼쪽 y-축: Thinking Presence ―――
ax1.set_xlabel('Step')
ax1.set_ylabel('Thinking Presence', color='tab:blue')
line1 = ax1.plot(step, thinking_presence, color='tab:blue', label='Thinking Presence')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# ――― 오른쪽 y-축: Accuracy ―――
ax2 = ax1.twinx()                        # 두 번째 y-축 생성
ax2.set_ylabel('Accuracy', color='tab:red')
line2 = ax2.plot(step, accuracy, color='tab:red', label='Accuracy')
ax2.tick_params(axis='y', labelcolor='tab:red')

# ――― 공통 타이틀 & 범례 ―――
plt.title('Thinking Progress')
# 두 축의 라인 객체를 합쳐서 하나의 범례로 표시
lines = line1 + line2
labels = [l.get_label() for l in lines]
fig.legend(lines, labels, loc='upper left', bbox_to_anchor=(0.1, 1.0))

fig.tight_layout()                       # 레이아웃 자동 조정
plt.savefig('./search_r1_explore/figs/thinking_progress.png')
plt.close()


# import pdb; pdb.set_trace()
