import os

from matplotlib.pyplot import plot

from model import*
from helper_function import *
from pre_processing_data import *


name_exp  = "train_fluorescence_test"

path_data = "./dataset_path/"
data_train_name = "imgs_train.hdf5"
data_mask_name =  "mask_train.hdf5"




N_epochs = 40
batch_size = 32

patch_height = 48
patch_width  = 48
sub_train    = 9000
inside_FOV = False

#Load Train and Mask Data
train_data = load_hdf5(path_data+data_train_name)
mask_train  = load_hdf5(path_data+data_mask_name)

#preprocessing data and scale to 0-1
train_data_process = pre_process(train_data)
mask_train = mask_train/255

data_consistency_check(train_data_process, mask_train)

# check masks are within 0-1
assert (np.min(mask_train) == 0 and np.max(mask_train) == 1)

print("\ntrain images/masks shape:")
print(train_data_process.shape,"/", mask_train.shape)
print("train images range (min-max): " + str(np.min(train_data_process)) + ' - ' + str(np.max(train_data_process)))
print("train masks are within 0-1\n")

patches_train, patches_mask = extract_random(train_data_process, mask_train, patch_height, patch_width, sub_train, inside_FOV)
data_consistency_check(patches_train, patches_mask)

N_sample = min(patches_train.shape[0],40)

n_ch = patches_train.shape[1]
patch_height = patches_train.shape[2]
patch_width = patches_train.shape[3]
model = Unet(n_ch, patch_height, patch_width)  #the U-net model
print("Check: final output of the network:")
print(model.output_shape)

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
# plot(model, to_file='./'+name_exp+'/'+name_exp + '_model.png')   #check how the model looks like
json_string = model.to_json()
open('./'+name_exp+'/'+name_exp +'_architecture.json', 'w').write(json_string)


checkpointer = ModelCheckpoint(filepath='./'+name_exp+'/'+name_exp +'_best_weights.h5', verbose=1, monitor='val_loss', mode='auto', save_best_only=True) #save at each epoch if the validation decreased

print(patches_mask.shape)
patches_mask = masks_Unet(patches_mask)
print(patches_mask.shape)

model.fit(patches_train, patches_mask, epochs=N_epochs, batch_size=batch_size, verbose=1, shuffle=True, validation_split=0.1, callbacks=[checkpointer])
model.save_weights('./'+name_exp+'/'+name_exp +'_last_weights.h5', overwrite=True)