#-----IMPORT-----#
from itertools import chain
from random import randint
import numpy as np
import librosa
from scipy.io.wavfile import read
import librosa.display
import os
#-----IMPORT-----#

#--------------Set Parameter--------------#
PCM = 48000
fft_size = 2048                 # Frame length
hl = int(fft_size / 4)          # Frame shift length
hi = 250                        # Height of image
wi = 250 - 1                    # Width of image
F_max = 20000                   # Freq max
#--------------Set Parameter--------------#

dataset_num = 3000
N = 17
all_dataset = 44 # *2(ja,jp)
save = True

images = []
labels =[]

dataset_cnt = 0

while dataset_cnt < dataset_num:

    ValueErr = False

#--------------Make Random List(length = 88)--------------#

    t = []

    while len(t) < N:
        j = randint(1,all_dataset)
        t.append(j)
        t = list(set(t))
    t.sort()

    s_list = [0] * 44

    for i in t:
        s_list[i-1] = 1

    cnt = 0
    list88 = [0] * 88

    for i in s_list:
        if i == 1:
            j = randint(0,1)
            if j == 1:
                list88[cnt] = 1
                list88[cnt + 1] = 0
            else:
                list88[cnt] = 0
                list88[cnt + 1] = 1
        cnt += 2

    # print("answer_label : ",list88)
    
#--------------Make Random List(length = 88)--------------#

#--------------Make filename by list88--------------#
    audio_list = []

    for i,j in enumerate(list88):
        if j == 1:
            if i%2 == 0: #日本語
                i = int(i/2) + 1
                if len(str(i)) == 1:
                    l = "J0" + str(i)
                else:
                    l = "J" + str(i)
            else:#英語
                i = int(i/2) + 1
                if len(str(i)) == 1:
                    l = "E0" + str(i)
                else:
                    l = "E" + str(i)
            audio_list.append(l)

#--------------Make filename by list88--------------#



#--------------Make delay_list--------------#
    all_data = []
    delay_list = []
    audio_length_list = []

    for name in audio_list:
        PCM, data = read("audio/Sample_Audio/"+name+".wav")
        # delay_random_num = randint(0, 5) * 4800    #! random delay No DEBUG
        delay_random_num = randint(0, 5 * 4800)     #! random delay No DEBUG
        delay_list.append(delay_random_num)
        cut_offset_data = data[delay_random_num:]
        all_data.append(cut_offset_data)
        audio_length_list.append(len(cut_offset_data))

#--------------Make delay_list--------------#

#------------------Fill Zero----------------#
    max_audio_length = max(audio_length_list)

    result = np.zeros(max_audio_length,dtype = int)

    for data in all_data:
        n_empty = max_audio_length - len(data)
        empty_list = np.zeros(n_empty,dtype = int)
        long_data = list(chain(data,empty_list))
        result += long_data

#------------------Fill Zero----------------#


#------------------Delete------------------#

    delete_num = randint(0,250000)
    result = result[:len(result) - delete_num]

#------------------Delete------------------#

#-----------------cut list------------------
    c = True
    timeout_cnt = 0
    timeout_bool = False

    frames = len(result)

    while c:
        split_list = []
        n_split = randint(2,5)
        #-----timeout-----
        timeout_cnt += 1
        if timeout_cnt > 20:
            print("TIME OUT")
            timeout_bool = True
            break 
        #-----timeout-----
        for i in range(n_split - 1):
            split_list.append(randint(1,frames))
        split_list.sort()
        split_list.insert(0,0)
        split_list.append(frames)
        c = False
        for i in range(n_split):
            if ( split_list[i + 1] - split_list[i] ) <= 0.5 * 48000:
                c = True
    
    if timeout_bool:
        continue
#-----------------cut list------------------

#-----------------cut audio------------------
    split_list[-1] += 1
    split_audio = []
    same_alldata = []

    # print("n_split : ",n_split)

    for j in range(n_split):
        split_data = result[split_list[j]:split_list[j + 1]]
        n_empty = 48000 * 3 - len(split_data)
        try:
            empty_list = np.zeros(n_empty)
        except ValueError:
            # print("value Error (split_data is too large)")
            ValueErr = True
            break
        same_length_data = np.array(list(chain(split_data,empty_list)))
        same_alldata.append(same_length_data)
#---------------------------Make Audio end-----------------------------#

    if ValueErr:
        continue
    
    #--------------0-1--------------#
    same_alldata = np.array(same_alldata)
    same_alldata = same_alldata/(2**15)
    #--------------0-1--------------#

    # print("max",np.max(same_alldata))
    # print("min",np.min(same_alldata))


    for j in range(n_split - 1):
        mono_data = same_alldata[j]
        # print(len(mono_data))
        mono_data = mono_data[0:wi*hl]

#--------------STFT--------------#
        S = librosa.feature.melspectrogram(
            y = mono_data, sr = PCM, n_mels = hi, fmax = F_max, hop_length = hl, 
            win_length = fft_size, n_fft = fft_size)

        S_dB = librosa.power_to_db(S, ref = np.max)
#--------------STFT--------------#

        S_dB = np.flipud(S_dB)
        
        # print("shape (枚数, 縦, 横) : ",S_dB.shape)
        
        images.append(S_dB)  #* 画像に追加 
        labels.append(np.array(list88)) 

    dataset_cnt += 1
    print("dataset_cnt : ", dataset_cnt)

r = randint(10000,100000)


if save == True:
    os.mkdir("D:/DATASET/Dataset_" + str(dataset_num) + "_" + str(all_dataset * 2) + "in" + str(N) + "h" + str(r))

    np.save("D:/DATASET/Dataset_" + str(dataset_num) + "_" + str(all_dataset * 2) +  "in" + str(N) +  "h" + str(r) + "/images.npy",images)
    np.save("D:/DATASET/Dataset_" + str(dataset_num) + "_" + str(all_dataset * 2) +  "in" + str(N) +  "h" + str(r) + "/labels.npy",labels)

#*------------------------------------------------------------------------
