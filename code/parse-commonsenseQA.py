import jsonlines
import sys
import csv

expl = {}
with open(sys.argv[2], 'rb') as f:
    for item in jsonlines.Reader(f):
        expl[item['id']] = item['explanation']['open-ended']

with open(sys.argv[1], 'rb') as f:
    with open(sys.argv[3],'w') as wf:
        wfw = csv.writer(wf,delimiter=',',quotechar='"')
        wfw.writerow(['id','question','choice_0','choice_1','choice_2','choice_3','choice_4','label','human_expl_open-ended'])
        for item in jsonlines.Reader(f):
            label = -1
            if(item['answerKey'] == 'A'):
                label = 0
            elif(item['answerKey'] == 'B'):
                label = 1
            elif(item['answerKey'] == 'C'):
                label = 2
            elif(item['answerKey'] == 'D'):
                label = 3
            else:
                label = 4
            wfw.writerow([item['id'],item['question']['stem'],item['question']['choices'][0]['text'],item['question']['choices'][1]['text'],item['question']['choices'][2]['text'],item['question']['choices'][3]['text'],item['question']['choices'][4]['text'],label,expl[item['id']]])
