'''
with open('scores.txt') as f:
    content = f.readline()


content = content.split(",")
scores = list(map(float,content))
'''

def get_scores(name = 'scores.txt'):
    with open(name) as f:
        content = f.readline()
    content = content.split(",")
    scores = list(map(float,content))
    return scores
