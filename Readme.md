




# Auto-labeling
    Use CNN to extract features from videos and build a classifer 
    (e.g., Random Forest) to recognize human activities from vides streaming.
     
# Steps:
    1. install <= python3.7
    2. install tensorflow < 2.0 (1.15)
    3. download model: vgg_16.cpkt
    4. install tqdm, future, opencv-python
    5. create "video.txt" file
    6. create "out" directory
    6. python3.7 feature_extraction.py --video_list video.txt  --network vgg --framework tensorflow --output_path out/ --tf_model ./slim/vgg_16.ckpt


# Structure:
    data: input data
    features: extracted features 
        pcap: pcap features
        video: video features
            nets: nerual networks
            slim: pre-trained models
    models: available models
    out: output results
    

# Issues:


1. tensorflow < 2.0, which requires python3.7 or python3.6
2. download pretrained models from https://github.com/tensorflow/models/tree/master/research/slim
    such as: vgg_16_2016_08_28.tar.gz
            tar -xvf vgg_16_2016_08_28.tar.gz
            vgg_16.cpkt

3. python3.7 feature_extraction.py --video_list data/video.txt  --network vgg --framework tensorflow --output_path out/ --tf_model ./slim/vgg_16.ckpt
    --video_list: requires a txt file which lists all the full paths of the videos
    create an output  directory  by 'mkdir out'
  
 
