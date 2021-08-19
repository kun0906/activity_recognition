
----- DATASET
1. download h36m.zip
    https://drive.google.com/drive/folders/1c7Iz6Tt7qbaw0c1snKgcGOD-JGSzuZ4X
    Reference: 
        https://github.com/garyzhao/SemGCN/blob/master/data/README.md
    Note that: 
        h36m.zip already has 3d labels (i.e., coordinates).
---!        
2. pip3.7 install cdflib (not necessary)
---

2. generate data_3d_h36m.npz
    cd data/
    python3.7 prepare_data_h36m.py --from-archive h36m.zip   
    cd ..   
    
    Note that:
        prepare_data_h36m.py will parse 3d labels (coordinates) from h36m.zip and the project 3d to 2d.  
        (includes two outputs: 3d.npz and 2d.npz)
    Error: (higher version of h5py), h5py has no attribute values
        File "prepare_data_h36m.py", line 73, in <module>
        positions = hf['3D_positions'].value.reshape(32, 3, -1).transpose(2, 0, 1)
     
    solution: 
        positions = hf['3D_positions'][:].reshape(32, 3, -1).transpose(2, 0, 1)

-------TRAINING
    python run.py -e 80 -k cpn_ft_h36m_dbb -arc 3,3,3,3,3

-------EVALUATION
1. download the pretrained model
    mkdir checkpoint
    cd checkpoint
    wget https://dl.fbaipublicfiles.com/video-pose-3d/pretrained_h36m_cpn.bin
    cd ..  
    
2. download data_2d_h36m_cpn_ft_h36m_dbb.npz
    cd data
    wget https://dl.fbaipublicfiles.com/video-pose-3d/data_2d_h36m_cpn_ft_h36m_dbb.npz
    wget https://dl.fbaipublicfiles.com/video-pose-3d/data_2d_h36m_detectron_ft_h36m.npz
    cd ..

3. evaluate on h36m
    python3.7 run.py -k cpn_ft_h36m_dbb -arc 3,3,3,3,3 -c checkpoint --evaluate pretrained_h36m_cpn.bin





------INFERENCE
1. install detectron2 (used in 'inference')
    python3.7 -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
    https://detectron2.readthedocs.io/en/latest/tutorials/install.html
    
    Note: without GPU
   modify "python3.7/site-packages/detectron2/config/defaults.py" 
        # _C.MODEL.DEVICE = "cuda" 
        _C.MODEL.DEVICE = "cpu"
        
        https://github.com/facebookresearch/detectron2/issues/1360
    
    (not necessary)     
    install certfi
   Error: 
        ssl.SSLCertVerificationError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:1091)
    Solution:
        python3.7 -c "import sys; print('\n'.join(sys.path))"
   /Applications/Python\ 3.7/Install\ Certificates.command
   https://stackoverflow.com/questions/44649449/brew-installation-of-python-3-6-1-ssl-certificate-verify-failed-certificate/44649450
   https://stackoverflow.com/questions/42098126/mac-osx-python-ssl-sslerror-ssl-certificate-verify-failed-certificate-verify

2. inference
   # generate 2D keypoint predictions from videos
   cd inference
   # python3.7 infer_video_d2.py --cfg COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml --output-dir out_directory --image-ext mp4 input_directory
   # python3.7 infer_video_d2.py --cfg COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml --output-dir out --image-ext mp4 ../../data/demo
   python3.7 infer_video_d2.py --cfg COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml --output-dir out3d --image-ext mp4
   ../../data/data-clean/refrigerator python3.7 infer_video_d2.py --cfg COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml
   --output-dir out3d --image-ext mkv ../../data/data-clean/refrigerator

   # create customized dataset "data/data_2d_custom_myvideos.npz"
   cd ../data
   # python3.7 prepare_data_2d_custom.py -i ../inference/out3d -o myvideos
   python3.7 prepare_data_2d_custom.py -i ../inference/out3d -o 2d_keypoints cd ..

   # get 3d prediction
   (only for one video)
   # python3.7 run.py -d custom -k myvideos -arc 3,3,3,3,3 -c checkpoint --evaluate pretrained_h36m_detectron_coco.bin --render --viz-subject ask_time_1_1614904536_1.mp4 --viz-action custom --viz-camera 0 --viz-video ../data/demo/ask_time_1_1614904536_1.mp4 --viz-output output.mp4 --viz-size 6
   python3.7 run.py -d custom -k 2d_keypoints -arc 3,3,3,3,3 -c checkpoint --evaluate pretrained_h36m_detectron_coco.bin
   --render --viz-subject ask_time_1_1614904536_1.mp4 --viz-action custom --viz-camera 0 --viz-video
   ../data/demo/ask_time_1_1614904536_1.mp4 --viz-output output.mp4 --viz-size 6

   (multi-videos)
   python3.7 predict.py

# run.py

1. test_generator.next_epoch()
   the generator will augment the data by flip each video.
    1. For each video, it will pad zero to the 2d keypoint data. e.g., the input 2d keypoint is (1, 351, 17, 2).After
       padding, the input data will be (1, 593, 17, 2)
       (1, 351, 17, 2):
       1 is number of input data 351 is the number of frames of the video 17 is the number of 2d keypoints in each frame
       2 is the coordinates (x, y) of each keypoint

2. predicted_3d_pos = model_pos(inputs_2d)
   input_2d is (2, 593, 17, 2)
   predicted_3d_pos is (1, 351, 17, 3)

    2.2 self._forward_blocks(x)
        x: (2, 34, 593)-> 2 is batch_size, 34 is the in_dim, 593 is the number of channels (input channels)
        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))
        self.expand_conv(x): (2, 1024, 591)-> 1024 is out_dim (channels), 591 is the size for each channel
        
        x : (2, 1024, 591)
        for i in blocks (4 blocks):
            x = self.layers_conv[2 * i](x)))
        i=0: (2, 1024, 585)
        i=1: (2, 1024, 567)
        i=0: (2, 1024, 513)
        i=0: (2, 1024, 351)
    
    x = shrink(x)
    x: (2, 51, 351)
    x = view(x) # resize x 
    x: (2, 351, 17, 3)


    




