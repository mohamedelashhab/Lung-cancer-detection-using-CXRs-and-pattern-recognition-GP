from menpofit.aam import HolisticAAM
from pathlib import Path
import menpo.io as mio

# method to load a database
def load_dataset(path_to_images, crop_percentage, max_images=None):
    images = []
    # load landmarked images
    for i in mio.import_images(path_to_images, max_images=max_images, verbose=True):
        # crop image
        i = i.crop_to_landmarks_proportion(crop_percentage)
        
        # convert it to grayscale if needed
        if i.n_channels == 3:
            i = i.as_greyscale(mode='luminosity')
            
        # append it to the list
        images.append(i)
    return images

path_to_Dset= Path('/run/media/m.a.ashhab/1ECA933908590CB5/GP/ASM')
training_images = load_dataset(path_to_Dset / 'Train_R', 0.1)
from menpowidgets import visualize_images
visualize_images(training_images)
test_images = load_dataset(path_to_Dset / 'Test_R/Test_R', 0.5, max_images=5)
visualize_images(test_images)
from menpo.feature import imgfeature, igo

@imgfeature
def custom_double_igo(image):
    return igo(igo(image))
custom_double_igo(training_images[0]).view()
from menpofit.aam import HolisticAAM


aam = HolisticAAM(
    training_images,
    group='PTS',
    verbose=True,
    holistic_features=custom_double_igo, 
    diagonal=120, 
    scales=(0.5, 1.0)
)

print(aam)

aam.view_aam_widget()

from menpofit.aam import LucasKanadeAAMFitter

fitter = LucasKanadeAAMFitter(aam, n_shape=[6, 12], n_appearance=0.5)

from menpofit.fitter import noisy_shape_from_bounding_box

fitting_results = []

for i in test_images:
    # obtain original landmarks
    gt_s = i.landmarks['PTS'].lms
    
    # generate perturbed landmarks
    s = noisy_shape_from_bounding_box(gt_s, gt_s.bounding_box())
    
    # fit image
    fr = fitter.fit_from_shape(i, s, gt_shape=gt_s) 
    fitting_results.append(fr)
    
    print(fr)

from menpowidgets import visualize_fitting_result

visualize_fitting_result(fitting_results)

from menpofit.aam import PatchAAM
from menpo.feature import double_igo

patch_based_aam = PatchAAM(
    training_images,
    group='PTS',
    verbose=True)

print(patch_based_aam)

patch_based_aam.view_aam_widget()

fitter = LucasKanadeAAMFitter(patch_based_aam, 
                              n_shape=[3, 12], 
                              n_appearance=50)

from menpofit.fitter import noisy_shape_from_bounding_box

fitting_results = []

for i in test_images:
    # obtain original landmarks
    gt_s = i.landmarks['PTS'].lms
    
    # generate perturbed landmarks
    s = noisy_shape_from_bounding_box(gt_s, gt_s.bounding_box())
    
    # fit image
    fr = fitter.fit_from_shape(i, s, gt_shape=gt_s) 
    fitting_results.append(fr)
    
    print(fr)

    visualize_fitting_result(fitting_results)