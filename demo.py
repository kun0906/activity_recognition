
flg = False
if flg:
    import os
    cmd = 'python3.7 feature_extraction.py --video_list data/video.txt  --network vgg --framework tensorflow --output_path out/ --tf_model ./slim/vgg_16.ckpt'
    # eval(cmd)
    os.system(cmd)

import numpy as np

# f = 'out/play_music_1_1615410036_1_vgg.npy'
f = 'out/ask_time_1_1614904536_1_vgg.npy'
fs = [# 'out/camera2_mkv/ask_weather_4_1616183946_2_vgg.npy',
    'out/ask_time_1_1614904536_1_vgg.npy',
      'out/ask_time_1_1614904536_2_vgg.npy'
]
fs2 = [ 'out/output_mp4/ask_time_1_1614904536_1_vgg.npy',
        'out/output_mkv/ask_time_1_1614904536_2_vgg.npy',
        'out/output_mkv/ask_time_1_1614904536_2_vgg.npy',
        ]
for f1, f2 in zip(fs, fs2):
    print('\n')
    print(f1)
    data = np.load(f1)
    print(data.shape)
    # sum
    # data = np.sum(data, axis=0)
    for i in range(len(data)):
        print(f'frame_{i+1}: {data[i, :]}')
    # print(data/sum(data))
    # a = np.quantile(data/sum(data), [0, 0.1, 0.5, 0.9, 1])
    # print(a)

    # print(f2)
    # data = np.load(f2)
    # print(data.shape)
    # print(data)
    # b = np.quantile(data, [0, 0.1, 0.5, 0.9, 1])
    # print(b)
    # print([v2/v1 for v1, v2 in zip(a, b)])


f  =  'out/output_mp4/ask_time_1_1614904536_1_vgg.npy'
data

