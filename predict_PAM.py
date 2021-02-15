from tensorflow.python.keras.models import model_from_json

from model import*
from helper_function import *
from pre_processing_data import *



path_data = "./dataset_path/"
data_test_name = "imgs_test.hdf5"
mask_test_name =  "mask_test.hdf5"

name_exp  = "train_fluorescence_test"
path_experiment = './' +name_exp +'/'

full_Image_test = 3
group_visual    = 1
avg_mode = True

patch_height = 48
patch_width  = 48
stride_height = 4
stride_width = 4
assert (stride_height < patch_height and stride_width < patch_width)

test_image = load_hdf5(path_data+data_test_name)
test_masks  = load_hdf5(path_data+mask_test_name)

print("Test data shape: ", test_image.shape)

full_img_height = test_image.shape[2]
full_img_width = test_image.shape[3]

test_imgs = pre_process(test_image)
test_masks = test_masks / 255

data_consistency_check(test_imgs, test_masks)


#============ Load the data and divide in patches
patches_imgs_test = None
new_height = None
new_width = None
masks_test  = None
patches_masks_test = None
if avg_mode == True:
    patches_imgs_test, new_height, new_width, masks_test = get_data_testing_overlap(test_imgs, test_masks, full_Image_test, patch_height, patch_width, stride_height, stride_width)
else:
    patches_imgs_test, patches_masks_test = get_data_testing(test_imgs, test_masks, full_Image_test, patch_height, patch_width)

best_last = 'best'
#Load the saved model
model = model_from_json(open(path_experiment+name_exp +'_architecture.json').read())
model.load_weights(path_experiment+name_exp + '_'+best_last+'_weights.h5')

#Calculate the predictions
predictions = model.predict(patches_imgs_test, batch_size=64, verbose=2)
# predictions = model.predict(patches_imgs_test, batch_size=32, verbose=1)
print("predicted images size :")
print(predictions.shape)

pred_patches = pred_to_imgs(predictions, patch_height, patch_width, "original")
print(pred_patches.shape)

#========== Elaborate and visualize the predicted images ====================
pred_imgs = None
orig_imgs = None
gtruth_masks = None
if avg_mode == True:
    pred_imgs = recompone_overlap(pred_patches, new_height, new_width, stride_height, stride_width)# predictions
    orig_imgs = pre_process(test_image[0:pred_imgs.shape[0],:,:,:])    #originals
    gtruth_masks = masks_test  #ground truth masks
else:
    pred_imgs = recompone(pred_patches,13,12)       # predictions
    orig_imgs = recompone(patches_imgs_test,13,12)  # originals
    gtruth_masks = recompone(patches_masks_test,13,12)  #masks
# apply the DRIVE masks on the repdictions #set everything outside the FOV to zero!!

## back to original dimensions
orig_imgs = orig_imgs[:,:,0:full_img_height,0:full_img_width]
pred_imgs = pred_imgs[:,:,0:full_img_height,0:full_img_width]
gtruth_masks = gtruth_masks[:,:,0:full_img_height,0:full_img_width]
print("Orig imgs shape: " +str(orig_imgs.shape))
print("pred imgs shape: " +str(pred_imgs.shape))
print("Gtruth imgs shape: " +str(gtruth_masks.shape))

for i in range(full_Image_test):
    org  = np.transpose(orig_imgs,(0,2,3,1))  #corect format for imshow
    pred = np.transpose(pred_imgs,(0,2,3,1))  #corect format for imshow
    visualize(org[i, :, :, :], path_experiment + "Ori_%d"%i)  # .show()
    visualize(pred[i, :, :, :], path_experiment + "Pred_%d" % i)  # .show()
