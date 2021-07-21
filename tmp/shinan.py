import glob
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

rng = 42

np.set_printoptions(threshold=np.inf)


def make_confusion_matrix(cf, feat_type, key, exp_name,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    normalize:     If True, show the proportions for each category. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html

    title:         Title for the heatmap. Default is None.
    '''

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names) == cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        row_sum = np.sum(cf, axis=1)
        cf_row_sum = np.array([[value] * len(row_sum) for value in row_sum]).flatten()
        #         print(cf_row_sum)
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten() / cf_row_sum]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        # Accuracy is sum of diagonal divided by total observations
        accuracy = np.trace(cf) / float(np.sum(cf))

        # if it is a binary confusion matrix, show some more stats
        if len(cf) == 2:
            # Metrics for Binary Confusion Matrices
            precision = cf[1, 1] / sum(cf[:, 1])
            recall = cf[1, 1] / sum(cf[1, :])
            f1_score = 2 * precision * recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy, precision, recall, f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize == None:
        # Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks == False:
        # Do not show categories if xyticks is False
        categories = False

    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf / np.sum(cf, axis=1), annot=box_labels, fmt="", cmap=cmap, cbar=cbar, xticklabels=categories,
                yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)

    if title:
        plt.title(title)

    plt.savefig('../plots/' + feat_type + '_' + key + '_' + exp_name + '_pcap_concat.png', bbox_inches='tight', dpi=300)


def load_data(in_file):
    with open(in_file, 'rb') as f:
        data = pickle.load(f)
    return data


#
def load_dataset(feat_type, device="*", vidInte=False, vidType='mp4', concat=True):
    X = []
    y = []
    print(device)
    if feat_type == 'keypoint':
        for activity in act_label.keys():
            dir = f'{feature_path}{activity}/keypoint/'
            print(glob.glob(dir + '*.dat'))
            for ft_files in glob.glob(dir + '*.dat'):
                X_, y_ = load_data(ft_files)
                X.append(X_)
                y.append(y_)
        #         print(X)
        #         print(y)
        return X, y

    else:
        #         for activity in act_label.keys():
        dir = feature_path + device + '/*/*/pcap-netml/' + feat_type + '/'
        print(len(glob.glob(dir + '*.dat')))
        for ft_files in glob.glob(dir + '*.dat'):
            #             scaler = MinMaxScaler()
            X_, y_ = load_data(ft_files)
            X_ = X_.reshape(1, 36)
            #             print(X_)
            #             scaler.fit(X_)
            #             X_ = scaler.transform(X_)
            #             print(X_)
            y_ = y_[0]
            if not vidInte:
                X.append(X_)
            else:
                index = ft_files.split('/')[-1].split('_filtered.dat')[0].split('STATS_')[1]
                if concat:
                    if vidType == "mp4":
                        vid_dir = f'{vid_feature_path_mp4}'
                        vid_X = np.load(vid_dir + str(index) + '_1_vgg.npy')
                        x_li = []
                        for x in X_:
                            x = np.concatenate((x, vid_X))
                            x_li.append(x)
                    elif vidType == "mkv":
                        vid_dir = f'{vid_feature_path_mkv}'
                        vid_X = np.load(vid_dir + str(index) + '_2_vgg.npy')
                        x_li = []
                        for x in X_:
                            x = np.concatenate((x, vid_X))
                            x_li.append(x)
                    elif vidType == "both":
                        vid_X_1 = np.load(vid_feature_path_mp4 + str(index) + '_1_vgg.npy')
                        vid_X_2 = np.load(vid_feature_path_mkv + str(index) + '_2_vgg.npy')
                        x_li = []
                        for x in X_:
                            x = np.concatenate((x, vid_X_1, vid_X_2))
                            x_li.append(x)
                else:
                    if vidType == "mp4":
                        vid_dir = f'{vid_feature_path_mp4}'
                        vid_X = np.load(vid_dir + str(index) + '_1_vgg.npy')
                        x_li = []
                        for x in X_:
                            x = vid_X
                            x_li.append(x)
                    elif vidType == "mkv":
                        vid_dir = f'{vid_feature_path_mkv}'
                        vid_X = np.load(vid_dir + str(index) + '_2_vgg.npy')
                        x_li = []
                        for x in X_:
                            x = vid_X
                            x_li.append(x)
                    elif vidType == "both":
                        vid_X_1 = np.load(vid_feature_path_mp4 + str(index) + '_1_vgg.npy')
                        vid_X_2 = np.load(vid_feature_path_mkv + str(index) + '_2_vgg.npy')
                        x_li = []
                        for x in X_:
                            x = np.concatenate((vid_X_1, vid_X_2))
                            x_li.append(x)
                X_ = np.array(x_li)
                X.append(X_)
            y.append(y_)

        #     print(X)
        #     print(y)
        scaler = MinMaxScaler()
        X = np.concatenate(X, axis=0)
        scaler.fit(X)
        X = scaler.transform(X)
    #         print(X)
    return np.array(X), np.array(y)


act_label = {"no_interaction": 0,
             "open_close_fridge": 1,
             "put_back_item": 2,
             "screen_interaction": 3,
             "take_out_item": 4}

classifiers = {"OvRLogReg": OneVsRestClassifier(LogisticRegression(random_state=rng)),
               "DecTree": DecisionTreeClassifier(random_state=rng),
               "LogReg": LogisticRegression(random_state=rng),
               "OvOSVC": OneVsOneClassifier(SVC(random_state=rng)),
               # #                OneVsOneClassifier(NuSVC(random_state=42)),
               "OvOGP": OneVsOneClassifier(GaussianProcessClassifier(random_state=rng)),
               "OvRLinearSVC": OneVsRestClassifier(LinearSVC(random_state=rng)),
               "OvRGP": OneVsRestClassifier(GaussianProcessClassifier(random_state=rng)),
               }


#
# # act_label = {"frg_no_interaction": 0,
# #              "frg_open_close_fridge": 1,
# #              "frg_put_back_item": 2,
# #              "frg_screen_interaction": 3,
# #              "frg_take_out_item": 4,
# #              "alx_no_interaction": 5,
# #              "alx_ask_weather": 6,
# #              "alx_play_music": 7,
# #              "nstc_no_interaction": 8,
# #              "nstc_ask_time": 9,
# #              "nstc_trigger_motion_detection": 10}
# feat_types = ['STATS']
# # feat_types = ['IAT','SIZE','IAT_SIZE','SAMP_NUM','SAMP_SIZE','STATS',
# #                   'FFT-IAT','FFT-IAT_SIZE','FFT-SIZE','FFT-SAMP_NUM', 'FFT-SAMP_SIZE']
# from warnings import filterwarnings
#
# filterwarnings('ignore')
#
# for feat in feat_types:
#     print(feat)
#     # eval(feat, 'mp4+netml',"refrigerator", True, "mp4")
#
#     eval(feat, 'mp4', "refrigerator", True, "mp4", False)


def eval(feat_type, exp, device="*", vid=False, vidType="mp4", concat=True):
    print(exp)
    # X, y = load_dataset(feat_type, device, vidInte=vid, vidType=vidType, concat=concat)
    X, y = gen_video_data(in_dir)
    df = pd.DataFrame(X)
    #     for k in range(0, len(X)):
    #         print(X[k][17], X[k][35], y[k], y[k])
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.33)
    #     print(X, y)
    kf = KFold(3, True, rng)
    for key in classifiers:
        print(key)
        for train_index, test_index in kf.split(X, y):
            #             print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            classifier = classifiers[key].fit(X_train, y_train)
            #         print(classifier.feature_importances_)

            predicted_labels = classifier.predict(X_test)
            cf_matrix = confusion_matrix(y_test, predicted_labels)
            size = len(act_label.keys())
            make_confusion_matrix(cf_matrix, feat_type, key, exp,
                                  figsize=(size * 1.5, size * 1.5),
                                  categories=act_label.keys(),
                                  cmap='Blues')
            #         disp = plot_confusion_matrix(classifier, X_test, y_test,
            #                                      cmap=plt.cm.Blues)
            plt.show()
    return df


def main():
    vid_feature_path_mp4 = "../../video-feature-clean/output_mp4/"
    vid_feature_path_mkv = "../../video-feature-clean/output_mkv/"
    #
    # X = []
    # f = []
    # y = []
    # act_label = {"frg_no_interaction": 0,
    #              "frg_open_close_fridge": 1,
    #              "frg_put_back_item": 2,
    #              "frg_screen_interaction": 3,
    #              "frg_take_out_item": 4}
    # # act_label = {"frg_no_interaction": 0,
    # #              "frg_open_close_fridge": 1,
    # #              "frg_put_back_item": 2,
    # #              "frg_screen_interaction": 3,
    # #              "frg_take_out_item": 4,
    # #              "alx_no_interaction": 5,
    # #              "alx_ask_weather": 6,
    # #              "alx_play_music": 7,
    # #              "nstc_no_interaction": 8,
    # #              "nstc_ask_time": 9,
    # #              "nstc_trigger_motion_detection": 10}
    #
    # for file in glob.glob("../../video-feature-clean/output_mp4/*.npy"):
    #     X.append(np.load(file))
    #     f.append(file)
    #     y.append(label)
    #     print(file)
    #     print(np.load(file).shape)
    #
    # from sklearn.model_selection import KFold

    feat = ''
    eval(feat, 'mp4', "refrigerator", True, "mp4", False)


if __name__ == '__main__':
    main()
