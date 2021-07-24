

# classical
- classical machine learning models to obtain the preliminary results

# cnn_bert 
-  one potential model for activity recognition
    1. python   ==  3.7.9
    2. install all the libraries in the requirement.txt
    3. download datasets (e.g, hmdb51) and weights (from https://onedrive.live.com/?authkey=%21ACfd54VbdRBn6qM&id=463D936354E78FA2%21107&cid=463D936354E78FA2)
    4. download 'resnext-101-64f-kinetics.pth' (find the link from models/cnn_bert/README.md) and put it into the models/cnn_bert/weights (if not exist, create a new folder)
    5. run the code (e.g., two_stream_bert2.py) under the "activity_recognition/"
         PYTHONPATH=./:../:../../ python3 models/cnn_bert/two_stream_bert2_main.py
    
        
    
    
    
reference:
    Kalfaoglu, M. Esat, Sinan Kalkan, and A. Aydin Alatan. "Late temporal modeling in 3d cnn architectures with bert for action recognition." In European Conference on Computer Vision, pp. 731-747. Springer, Cham, 2020.
    2020
    3D-CNN+Transformer(BERT)
    https://github.com/artest08/LateTemporalModeling3DCNN



