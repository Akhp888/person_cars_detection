"""
    Python main script for plotting metrics of object detction groud truth vs predicted set
"""
import os
import numpy as np
import torch
import glob
from PIL import Image

from metrics_od.utils.general import xywh2xyxy,xyxy2xywh
from metrics_od.utils.metrics import ConfusionMatrix


def xyxy2cord(loaded_tensor,gn_x):
    """
    function should convert the orginal xyxy to xywh tensor

    :param loaded_tensor: tensor of the label file , [tensor]
    :param gn_x: 4 element list of dimensions for normalization , [list]

    :return tbox_full_pred: xyxyx to xywh converted tensor, [tensor]
    
    """
    A= [np.nan,np.nan,np.nan,np.nan]
    for (cls, *xyxy) in loaded_tensor.tolist():
        new_value = (xywh2xyxy(torch.tensor([xyxy[1:]]).view(1, 4)) * gn_x).view(-1).tolist()
        A = np.vstack((A, new_value))
    tbox = torch.tensor(A[1:,:])
    tbox_full_pred = torch.cat((loaded_tensor[:, 1:2], tbox), 1)
    return tbox_full_pred

def txt2tensor(label_full_path):
    """
    function should should consume a label txt file to generate tensor 

    :param label_full_path: path to the txt file, [string]

    :return loaded_tensor: tensor of the txt file, [tensor]
    
    """
    label_txt = (np.loadtxt(label_full_path, delimiter=" "))
    if len(label_txt) == 0:
        z = np.zeros((1,6))
        label_txt = z
    elif len(label_txt.shape) == 1:
        z_size = 1
        z = np.zeros((z_size))
        label_txt = np.append(z , label_txt)
        label_txt = [label_txt]
    else:
        z_size = label_txt.shape[0]
        z = np.zeros((z_size,1))
        label_txt = np.append(z , label_txt, axis=1)
    
    loaded_tensor = torch.tensor(label_txt)
    return loaded_tensor


def metrics_od(path_true_label,path_predicted_label,path_output_dir,class_labels,iou=0.45,image_ext='jpg'):
    """
    function should plot the metric graphs for the input sets of labels

    :param path_true_label: path to true label directory, [string]
    :param path_predicted_label: path to predicted label directory, [string]
    :param path_output_dir: path to save graphs, [string]
    :param class_labels: class names in same order as list, [list]
    :param iou: iou threshold, [float]
    :param image_ext: extension of true value image if exists in true label directory , [string]
    
    """
    nc= len(class_labels)

    list_predicted = [os.path.basename(x) for x in (glob.glob(os.path.join(path_predicted_label,'*.txt')))]
    list_actual = [os.path.basename(x) for x in (glob.glob(os.path.join(path_true_label,'*.txt')))]
    list_predicted_new_added = set(list_actual)-set(list_predicted)
    list_actual_new_added = set(list_predicted)-set(list_actual)

    output_path_check = os.path.isdir(path_output_dir)
    input_path_check = os.path.isdir(path_predicted_label) and os.path.isdir(path_true_label)
    if output_path_check == False:
        print('output_dir path doesnt exist')
    if input_path_check == False:
        print('Check path for True and predicted label directory')
    if (output_path_check and input_path_check) == True:
        for f in list_predicted_new_added:
            with open((os.path.join(path_predicted_label,f)), 'w') as fp:
                pass

        for f in list_actual_new_added:
            with open((os.path.join(path_true_label,f)), 'w') as fp:
                pass

        all_list_labels_predicted = [x for x in (glob.glob(os.path.join(path_predicted_label,'*.txt')))]

        confusion_matrix = ConfusionMatrix(nc=nc, iou_thres=iou)

        for f in all_list_labels_predicted:
            loaded_tensor_pred = txt2tensor(f)
            # print('predicted_values',loaded_tensor_pred)

            _, tail = os.path.split(f)
            true_label_full_path = os.path.join(path_true_label,tail)

            loaded_tensor_true = txt2tensor(true_label_full_path)
            # print('True_values',loaded_tensor_true)

            if os.path.isfile(os.path.join(path_true_label,tail[:-3]+image_ext)):
                im = Image.open(os.path.join(path_true_label,tail[:-3]+image_ext))
                width, height = im.size
                gn_x = torch.tensor([width,height,width,height])
            else:
                gn_x = torch.tensor([100,100,100,100])

            tbox_full_true = xyxy2cord(loaded_tensor_true,gn_x)
            # print('tbox_True',tbox_full_true)

            tbox_full_pred = xyxy2cord(loaded_tensor_pred,gn_x)
            shape_x = loaded_tensor_pred.shape[0]
            tbox_full_pred = torch.cat((tbox_full_pred[:,1:5], torch.tensor(np.ones((shape_x,1))) ,tbox_full_pred[:,:1]),1)
            # print('tbox_pred',tbox_full_pred)

            confusion_matrix.process_batch(tbox_full_pred, tbox_full_true)

        confusion_matrix.plot(save_dir=path_output_dir, names=class_labels)
        print('confusion matrix plotted')

def tf_2_yolo(tf_label_full_path, output_file_path, image_ext='jpg'):
    """
    function should should consume a label txt file to generate tensor

    :param label_full_path: path to the tf prediction txt file, [string]
    :param output_file_path: path to save the yolo converted txt file, [string]
    :param image_ext: extension of true value image if exists in true label directory , [string]

    """
    input_path_check = os.path.isdir(tf_label_full_path) and os.path.isdir(output_file_path)
    if input_path_check == False:
        print('Check path for input label and results save directory')

    tf_label_full_list = [x for x in (glob.glob(os.path.join(tf_label_full_path,'*.txt')))]
    for f in tf_label_full_list:
        try:
            _, tail = os.path.split(f)
            with open(os.path.join(output_file_path,tail), 'a') as the_file:
                loaded_tensor = txt2tensor(f)
                loaded_tensor_swaped = loaded_tensor.clone().detach()
                loaded_tensor_swaped[:,[2, 3]] = loaded_tensor_swaped[:,[3, 2]]
                loaded_tensor_swaped[:,[4, 5]] = loaded_tensor_swaped[:,[5, 4]]

                for (cls, *xyxy) in loaded_tensor_swaped.tolist():
                    new_value = (xyxy2xywh((torch.tensor(xyxy[1:])).view(1, 4))).view(-1).tolist()
                    the_file.write(str(int(xyxy[0:1][0])))
                    for values in new_value:
                        the_file.write(' ')
                        the_file.write("{:f}".format(values))
                    the_file.write('\n')
        except Exception as e:
            print(e)
            pass

if __name__ == '__main__':

    print('analysing')
    path_true_label = 'D:/projects/misc/ev_inference/trainval/images'
    path_predicted_label = 'D:/projects/misc/ev_inference/trainval/predictions/labels'
    path_output_dir = '.'
    class_labels = ['person','car']
    iou = 0.3
    metrics_od(path_true_label,path_predicted_label,path_output_dir,class_labels,iou)
