import  os, glob
import  random, csv

def load_csv(root, filename, name2label):
    # root:数据集根目录
    # filename:csv文件名
    # name2label:类别名编码表
    if not os.path.exists(os.path.join(root, filename)):
        images = []
        for name in name2label.keys():
            # 'pokemon\\mewtwo\\00001.png
            images += glob.glob(os.path.join(root, name, '*.png'))
            images += glob.glob(os.path.join(root, name, '*.jpg'))
            images += glob.glob(os.path.join(root, name, '*.jpeg'))

        # 1167, 'pokemon\\bulbasaur\\00000000.png'
        print(len(images), images)

        random.shuffle(images)
        with open(os.path.join(root, filename), mode='w', newline='') as f:
            writer = csv.writer(f)
            for img in images:  # 'pokemon\\bulbasaur\\00000000.png'
                name = img.split(os.sep)[-2]
                label = name2label[name]
                # 'pokemon\\bulbasaur\\00000000.png', 0
                writer.writerow([img, label])
            print('written into csv file:', filename)

    # read from csv file
    images, labels = [], []
    with open(os.path.join(root, filename)) as f:
        reader = csv.reader(f)
        for row in reader:
            # 'pokemon\\bulbasaur\\00000000.png', 0
            img, label = row
            label = int(label)

            images.append(img)
            labels.append(label)

    assert len(images) == len(labels)

    return images, labels


def load_pokemon(root, mode='train'):
    # 创建数字编码表
    name2label = {}  # "sq...":0
    for name in sorted(os.listdir(os.path.join(root))):
        if not os.path.isdir(os.path.join(root, name)):
            continue
        # 给每个类别编码一个数字
        name2label[name] = len(name2label.keys())

    # 读取Label信息
    # [file1,file2,], [3,1]
    images, labels = load_csv(root, 'images.csv', name2label)

    if mode == 'train':  # 60%
        images = images[:int(0.6 * len(images))]
        labels = labels[:int(0.6 * len(labels))]
    elif mode == 'val':  # 20% = 60%->80%
        images = images[int(0.6 * len(images)):int(0.8 * len(images))]
        labels = labels[int(0.6 * len(labels)):int(0.8 * len(labels))]
    else:  # 20% = 80%->100%
        images = images[int(0.8 * len(images)):]
        labels = labels[int(0.8 * len(labels)):]

    return images, labels, name2label