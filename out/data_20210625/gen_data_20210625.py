import os

cmd = 'python3.7 feature_extraction.py --video_list data/video.txt  --network vgg --framework tensorflow --output_path out/ --tf_model ./slim/vgg_16.ckpt'
# eval(cmd)
os.system(cmd)
