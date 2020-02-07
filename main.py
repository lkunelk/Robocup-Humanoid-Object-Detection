import cv2

data_path = '/home/nam/darknet/data/bitbot/imageset_159/'
result_path = '/home/nam/darknet/'
results_file = 'results_159.txt'
list_file = 'test_159.list'

labels = {}
detection = {}
name_ind, x1_ind, x2_ind, y1_ind, y2_ind = 1, 4, 6, 5, 7

def show_image(path):
    global labels, detection
    name = path.split('/')[-1]
    img = cv2.imread(path, cv2.IMREAD_COLOR)

    points = labels[name]
    img = cv2.rectangle(img, points[0], points[1], (0, 0, 255), 1)

    dets = detection.get(name, [])
    for det in dets:
        img = cv2.rectangle(img, det[1], det[2], (0, 255, 0), 2)

    cv2.imshow('name', img)
    cv2.waitKey(0)

with open(data_path + 'labels.txt') as file:
    for line in file:
        if 'label::ball' in line:
            tokens = line.split('|')
            name = tokens[name_ind]
            x1, x2, y1, y2 = map(int, [tokens[x1_ind], tokens[x2_ind], tokens[y1_ind], tokens[y2_ind]])
            if not name in labels:
                labels[name] = ((x1, y1), (x2, y2))
            else:
                pass
                #print('repeated label:', name)

with open(result_path + results_file) as file:
    path = ''
    name = ''
    for line in file:
        if 'Image Path:' in line:
            tokens = line.split(': ')
            path = tokens[1]
            name = path.split('/')[-1]
            #print(name)
        elif 'sports ball:' in line:
            tokens = line.split()
            x1, x2, y1, y2 = map(int, tokens[-4:])
            prob = int(tokens[-5][:-1])
            #print(prob, x1, x2, y1, y2)
            if name not in detection:
                detection[name] = []
            detection[name].append((prob, (x1, y1), (x2, y2)))

with open('/home/nam/darknet/data/bitbot/' + list_file) as file:
    total = 0
    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0
    for path in sorted(file):
        name = path.split('/')[-1][:-1]
        if name in detection and name in labels:
            true_pos += 1
        elif name in detection and name not in labels:
            false_pos += 1
        elif name not in detection and name not in labels:
            true_neg += 1
        elif name not in detection and name in labels:
            false_neg += 1
        total += 1
    print('tp:', true_pos, ', fp:', false_pos, ', tn:', true_neg,', fn:', false_neg, ', total:', total)