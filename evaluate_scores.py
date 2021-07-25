import pandas as pd
import os
import glob
import subprocess
import tqdm
import pandas as pd
import glob
import itertools
import shutil
import time
import re
import cv2
import numpy as np

import matplotlib.patches as patches
import matplotlib.pyplot as plt

def create_bbs_from_txt(txt_file):
    '''
    Reads yolo label file and stores bounding boxes in a DataFrame
    :param txt_file: Label file
    :return: DataFrame
    '''
    # using pandas for reading the txt file to get no problems with the varying whitespace separations
    try:
        boxes_df = pd.read_table(txt_file, header=None, delim_whitespace=True)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    boxes_df.rename(columns={0:'class',1:'cx',2:'cy',3:'w',4:'h',5:'conf'},inplace=True)
    return boxes_df

def index_where(sr):
    if (sr == True).any():
        return sr.loc[sr == True].index.tolist()
    return []

def eval_performance(LABEL_PATH,RESULT_PATH,cls=0):
    if not os.path.exists(LABEL_PATH):
        raise RuntimeError("Label path \"{}\" not found".format(LABEL_PATH))
    if not os.path.exists(RESULT_PATH):
        raise RuntimeError("Results path \"{}\" not found".format(RESULT_PATH))

    missing = 0
    total = 0
    all_ao = []
    results = []
    for f in tqdm.tqdm(sorted(list( glob.glob(LABEL_PATH+'/*.jpg') ))):
        txt = os.path.join(LABEL_PATH, os.path.splitext(os.path.split(f)[1])[0] + '.txt')
        g = os.path.join(RESULT_PATH,os.path.splitext(os.path.split(f)[1])[0] + '.txt')

        if not os.path.exists(txt) or not os.path.getsize(txt):
            if os.path.exists(g):
                actual = create_bbs_from_txt(g)
                results.append({'name':os.path.split(f)[1],'num_norm':0,'num_found':actual.shape[0],'overlaps':[], 'unmatched_detected':[],
                                'unmatched_labeled':[]})
        else:
            norm = create_bbs_from_txt(txt)
            norm['unmatched'] = True
            norm = norm.loc[norm['class'] == cls]
            total += norm.shape[0]

            if os.path.exists(g) and os.path.getsize(g):
                res = []
                actual = create_bbs_from_txt(g)
                actual['unmatched'] = True
                actual = actual.loc[actual['class'] == cls]

                for n_index,(nx,ny,nw,nh,*other) in norm[['cx','cy','w','h']].iterrows():

                    # find potential match candidates by overlaping area that is positive
                    candidates = []
                    for a_index,(ax,ay,aw,ah,*other) in actual[['cx','cy','w','h']].iterrows():
                        a = min([nx+nw/2,ax+aw/2]) - max([nx-nw/2,ax-aw/2])
                        b = min([ny+nh/2,ay+ah/2]) - max([ny-nh/2,ay-ah/2])
                        if a > 0 and b > 0:
                            candidates.append({'overlap':a*b,'index':a_index})

                    # take best ie largest overlap
                    if len(candidates):
                        best = pd.DataFrame(candidates).sort_values(['overlap'],ascending=False).iloc[0]
                        o = best['overlap']
                        r = o/(nw*nh)
                        a_index = best['index']
                        # at least 50% overlap, and only match once
                        if r > 0.3 and actual.loc[a_index,'unmatched']:
                            actual.loc[a_index,'unmatched'] = False
                            norm.loc[n_index,'unmatched'] = False
                            res.append((o,r))
                            all_ao.append((o,r))


                results.append({'name':os.path.split(f)[1],'num_norm':norm.shape[0],'num_found':actual.shape[0],'overlaps':[r for _,r in res],
                                'unmatched_detected':index_where(actual['unmatched']),
                                'unmatched_labeled':index_where(norm['unmatched'])})
                missing += norm.shape[0] - len(res)
            else:
                results.append({'name':os.path.split(f)[1],'num_norm':norm.shape[0],'num_found':0,'overlaps':[],
                                'unmatched_detected':[],
                                'unmatched_labeled':index_where(norm['unmatched'])})
                missing += norm.shape[0]

    df = pd.DataFrame(results)
    #     print(df)
    df['TP'] = df['overlaps'].apply(len)
    df['FN'] = df['num_norm'] - df['TP']
    df['FP'] = df['num_found'] - df['TP']
    df['lbs'] = 1
    return df

def plot_overview(all_performance,cls=['joint','pad'],plot_name='plot'):

    overview = pd.DataFrame([v[['num_norm','TP','FP','FN', 'lbs']].sum() for k,v in all_performance.items()])
    overview.index = pd.MultiIndex.from_tuples(all_performance.keys(),names=['Class','set'])
    overview['FP rate'] = overview['FP']/overview['num_norm']
    overview['FN rate'] = overview['FN']/overview['num_norm']
    overview['TP rate'] = overview['TP']/overview['num_norm']
    print(overview)


    sizes = overview.index.get_level_values(1).unique()

    fig,ax = plt.subplots(1,3,figsize=(20,6))
    ax = ax.reshape((1,3))

    for i in overview.index.get_level_values(0).unique():
        print(i)
        (overview.loc[i]['TP rate']*100).plot(ax=ax[0,0],c="rgbcmyk"[i])
        (overview.loc[i]['FP rate']*100).plot(ax=ax[0,1],c="rgbcmyk"[i])
        (overview.loc[i]['FN rate']*100).plot(ax=ax[0,2],c="rgbcmyk"[i])

    ax[0,0].axhline(y=overview['TP rate'].mean()*100, color='b', linestyle='--')
    ax[0,1].axhline(y=overview['FP rate'].mean()*100, color='b', linestyle='--')
    ax[0,2].axhline(y=overview['FN rate'].mean()*100, color='b', linestyle='--')
    ax[0,0].set_title("TP rate")
    ax[0,1].set_title("FP rate")
    ax[0,2].set_title("FN rate")

    for i in range(3):
        ax[0,i].legend([cls,'mean'])
    #         ax[0,i].set_xticks(sizes)

    #     ax[0,0].set_yticks(np.arange(70,115,5.0))
    #     ax[0,1].set_yticks(np.arange(0,8,0.5))
    #     ax[0,2].set_yticks(np.arange(0,30,1))

    plt.savefig(plot_name)
    plt.show()

def plot_overview_one(all_performance,cls=['joint','pad']):

    overview = pd.DataFrame([v[['num_norm','TP','FP','FN']].sum() for k,v in all_performance.items()])
    print(all_performance.keys())
    overview.index = pd.MultiIndex.from_tuples(all_performance.keys(),names=['Class','set'])
    overview['FP rate'] = overview['FP']/overview['num_norm']
    overview['FN rate'] = overview['FN']/overview['num_norm']
    overview['TP rate'] = overview['TP']/overview['num_norm']

    sizes = overview.index.get_level_values(1).unique()

    fig,ax = plt.subplots(1,3,figsize=(20,6))
    ax = ax.reshape((1,3))


    (overview.loc[i]['TP rate']*100).plot(ax=ax[0,0],c="rgbcmyk"[i],label="mean")
    (overview.loc[i]['FP rate']*100).plot(ax=ax[0,1],c="rgbcmyk"[i], label="mean")
    (overview.loc[i]['FN rate']*100).plot(ax=ax[0,2],c="rgbcmyk"[i], label="mean")

    ax[0,0].set_title("TP rate")
    ax[0,1].set_title("FP rate")
    ax[0,2].set_title("FN rate")

    for i in range(3):
        ax[0,i].legend(cls)
        ax[0,i].set_xticks(sizes)

    ax[0,0].set_yticks(np.arange(80,101.0,1.0))
    ax[0,1].set_yticks(np.arange(0,20.5,0.5))
    ax[0,2].set_yticks(np.arange(0,20.5,0.5))


    plt.show()

import os
import cv2
import matplotlib.pyplot as plt
import imagesize


def crop_box(img_cv,dw,dh,x, y, w, h):

    l = int((x - w / 2) * dw)
    r = int((x + w / 2) * dw)
    t = int((y - h / 2) * dh)
    b = int((y + h / 2) * dh)

    if l < 0:
        l = 0
    if r > dw - 1:
        r = dw - 1
    if t < 0:
        t = 0
    if b > dh - 1:
        b = dh - 1

    tl = (l, t)
    br = (r, b)

    return img_cv[tl[1]:br[1], tl[0]:br[0]]

def plot_label_vs_detections(df,P,Q,save_path):
    for name,u_det,u_lab in zip(df['name'],df['unmatched_detected'],df['unmatched_labeled']):
        im = cv2.imread(os.path.join(P,name))

        ntxt = os.path.join(P,os.path.splitext(name)[0]+'.txt')
        atxt = os.path.join(Q,os.path.splitext(name)[0]+'.txt')

        if not os.path.exists(ntxt):
            print("Normative labels {} not found, skip")
            continue

        n = create_bbs_from_txt(ntxt)
        if not n.shape[0]:
            print("WARNING Labelfile \"{}\" exists but is empty!".format(ntxt))
            n = None
        else:
            n['cx'] *= im.shape[1]
            n['cy'] *= im.shape[0]
            n['w'] *= im.shape[1]
            n['h'] *= im.shape[0]

        a = None
        if os.path.exists(atxt):
            a = create_bbs_from_txt(atxt)
            if not a.shape[0]:
                print("WARNING Labelfile \"{}\" exists but is empty!".format(atxt))
                a = None
            else:
                a['cx'] *= im.shape[1]
                a['cy'] *= im.shape[0]
                a['w'] *= im.shape[1]
                a['h'] *= im.shape[0]


        imratio = im.shape[0]/im.shape[1]
        fig,ax = plt.subplots(2,1,figsize=(100,40))

        ax[0].imshow(im,aspect='equal',cmap='gray')
        ax[1].imshow(im,aspect='equal',cmap='gray')

        # Attention: for some reason the integer column class gets cast to float when using iterrows and tuple unpack
        if n is not None:
            for i,(c,cx,cy,w,h,*other) in n[['class','cx','cy','w','h']].iterrows():
                ax[0].add_patch(patches.Rectangle((cx-w/2, cy-h/2), w, h, linewidth=1, edgecolor="gr"[1], facecolor='none'))
                #                 ax[0] = crop_box(ax[0], w, h, float(cx) ,float(cy) ,float(cx)+0.06 ,float(cy)+0.12)
                if i in u_lab:
                    ax[0].text(cx-w/2,cy,"NOT MATCHED",color="gr"[1])

        if a is not None:
            for i,(c,cx,cy,w,h,*other) in  a[['class','cx','cy','w','h']].iterrows():
                ax[1].add_patch(patches.Rectangle((cx-w/2, cy-h/2), w, h, linewidth=1, edgecolor="gr"[1], facecolor='none'))
                #                 ax[1] = crop_box(ax[1], w, h, float(cx) ,float(cy) ,float(cx)+0.06 ,float(cy)+0.12)
                if i in u_det:
                    ax[1].text(cx-w/2,cy,"NOT MATCHED",color="gr"[1])

        ax[0].set_title('Normative')
        ax[1].set_title('Detection')
        plt.suptitle(name)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path,os.path.splitext(name)[0]+'.jpg'))
#         plt.clf()
#         plt.show()

def print_res(df,imsize):
    print("Image Size {:d}:".format(imsize))
    r = df[['num_norm','TP','FP','FN']].sum()
    print("Total number of labels: {:d}".format(r['num_norm']))
    print("TP {:8d} rate: {:.2f}".format(r['TP'],100.0*r['TP']/r['num_norm']))
    print("FP {:8d} rate: {:.2f}".format(r['FP'],100.0*r['FP']/r['num_norm']))
    print("FN {:8d} rate: {:.2f}".format(r['FN'],100.0*r['FN']/r['num_norm']))

# since coco dataset by default has person as class 0 and car as class as 2 , and our dataset has cars as label 1 , we need to replace 2 with 1 in labels
txt_file_list = glob.glob('D:/projects/misc/ev_inference/trainval/predictions/labels/*.txt')
for txt_file in txt_file_list:
    with open(txt_file) as f:
        lines = f.readlines()
        for i in range(len(lines)):
            #             print(lines[i].replace('15','0'))
            lines[i] = lines[i].replace('2','1')

    with open(txt_file, "w") as f:
        f.writelines(lines)

classes=[0 , 1]

all_performance = {}
complete_df = pd.DataFrame([])
for cls in classes:
    df= eval_performance('D:/projects/misc/ev_inference/trainval/images','D:/projects/misc/ev_inference/trainval/predictions/labels',cls)
    all_performance[(cls)] = df
    complete_df = complete_df.append(df)



with open('D:/projects/misc/ev_inference/trainval/data_splits/train.txt', "w") as f:
    for txt_file in txt_file_list:
        f.writelines(txt_file)
        f.writelines('\n')

t = complete_df.sort_values(['FN'],ascending=False)[:10]
t = t.loc[t['FN']>0]
if t.shape[0] > 0:
    plot_label_vs_detections(t,'D:/projects/misc/ev_inference/trainval/images','D:/projects/misc/ev_inference/trainval/predictions/labels','D:/projects/misc/ev_inference/trainval/inference/FN')

t = complete_df.sort_values(['FP'],ascending=False)[:10]
t = t.loc[t['FP']>0]
if t.shape[0] > 0:
    plot_label_vs_detections(t,'D:/projects/misc/ev_inference/trainval/images','D:/projects/misc/ev_inference/trainval/predictions/labels','D:/projects/misc/ev_inference/trainval/inference/FP')