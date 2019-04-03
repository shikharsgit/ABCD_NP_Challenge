from sklearn.externals.joblib import Memory
from nilearn.image import load_img,smooth_img
from nilearn import datasets
from nilearn.input_data import NiftiLabelsMasker,NiftiMasker
from nilearn.regions import RegionExtractor
from nilearn import _utils
from nilearn._utils import logger, CacheMixin, _compose_err_msg
from nilearn._utils.class_inspect import get_params
from nilearn._utils.niimg_conversions import _check_same_fov
from nilearn import masking
from nilearn import image
from nilearn.input_data.base_masker import filter_and_extract, BaseMasker
import os
import pandas as pd
from joblib import Parallel, delayed
import joblib
# import multiprocessing
import glob

class _ExtractionFunctor(object):

    func_name = 'nifti_labels_masker_extractor'

    def __init__(self, _resampled_labels_img_, background_label):
        self._resampled_labels_img_ = _resampled_labels_img_
        self.background_label = background_label

    def __call__(self, imgs):
        from nilearn.regions import signal_extraction

        return signal_extraction.img_to_signals_labels(
            imgs, self._resampled_labels_img_,
            background_label=self.background_label)




class NiftiLabelsMasker2(BaseMasker, CacheMixin):

    def __init__(self, labels_img, background_label=0, mask_img=None,
                 smoothing_fwhm=None, standardize=False, detrend=False,
                 low_pass=None, high_pass=None, t_r=None, dtype=None,
                 resampling_target="data",
                 memory=Memory(cachedir=None, verbose=0), memory_level=1,
                 verbose=0):
        self.labels_img = labels_img
        self.background_label = background_label
        self.mask_img = mask_img

        # Parameters for _smooth_array
        self.smoothing_fwhm = smoothing_fwhm

        # Parameters for clean()
        self.standardize = standardize
        self.detrend = detrend
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.t_r = t_r
        self.dtype = dtype

        # Parameters for resampling
        self.resampling_target = resampling_target

        # Parameters for joblib
        self.memory = memory
        self.memory_level = memory_level
        self.verbose = verbose

        if resampling_target not in ("labels", "data", None):
            raise ValueError("invalid value for 'resampling_target' "
                             "parameter: " + str(resampling_target))

    def fit(self, X=None, y=None):
        """Prepare signal extraction from regions.
        All parameters are unused, they are for scikit-learn compatibility.
        """
        logger.log("loading data from %s" %
                   _utils._repr_niimgs(self.labels_img)[:200],
                   verbose=self.verbose)
        self.labels_img_ = _utils.check_niimg_3d(self.labels_img)
        if self.mask_img is not None:
            logger.log("loading data from %s" %
                       _utils._repr_niimgs(self.mask_img)[:200],
                       verbose=self.verbose)
            self.mask_img_ = _utils.check_niimg_3d(self.mask_img)
        else:
            self.mask_img_ = None

        # Check shapes and affines or resample.
        if self.mask_img_ is not None:
            if self.resampling_target == "data":
                # resampling will be done at transform time
                pass
            elif self.resampling_target is None:
                if self.mask_img_.shape != self.labels_img_.shape[:3]:
                    raise ValueError(
                        _compose_err_msg(
                            "Regions and mask do not have the same shape",
                            mask_img=self.mask_img,
                            labels_img=self.labels_img))
                if not np.allclose(self.mask_img_.affine,
                                   self.labels_img_.affine):
                    raise ValueError(_compose_err_msg(
                        "Regions and mask do not have the same affine.",
                        mask_img=self.mask_img, labels_img=self.labels_img))

            elif self.resampling_target == "labels":
                logger.log("resampling the mask", verbose=self.verbose)
                self.mask_img_ = image.resample_img(
                    self.mask_img_,
                    target_affine=self.labels_img_.affine,
                    target_shape=self.labels_img_.shape[:3],
                    interpolation="nearest",
                    copy=True)
            else:
                raise ValueError("Invalid value for resampling_target: " +
                                 str(self.resampling_target))

            mask_data, mask_affine = masking._load_mask_img(self.mask_img_)

        return self

    def fit_transform(self, imgs, confounds=None):
        """ Prepare and perform signal extraction from regions.
        """
        return self.fit().transform(imgs, confounds=confounds)

    def _check_fitted(self):
        if not hasattr(self, "labels_img_"):
            raise ValueError('It seems that %s has not been fitted. '
                             'You must call fit() before calling transform().'
                             % self.__class__.__name__)

    def transform_single_imgs(self, imgs, confounds=None):
        """Extract signals from a single 4D niimg.
        Parameters
        ----------
        imgs: 3D/4D Niimg-like object
            See http://nilearn.github.io/manipulating_images/input_output.html
            Images to process. It must boil down to a 4D image with scans
            number as last dimension.
        confounds: CSV file or array-like, optional
            This parameter is passed to signal.clean. Please see the related
            documentation for details.
            shape: (number of scans, number of confounds)
        Returns
        -------
        region_signals: 2D numpy.ndarray
            Signal for each label.
            shape: (number of scans, number of labels)
        """
        # We handle the resampling of labels separately because the affine of
        # the labels image should not impact the extraction of the signal.

        if not hasattr(self, '_resampled_labels_img_'):
            self._resampled_labels_img_ = self.labels_img
        if self.resampling_target == "data":
            imgs_ = _utils.check_niimg_4d(imgs)
            
            if not _check_same_fov(imgs_, self._resampled_labels_img_):
                if self.verbose > 0:
                    print("Resampling labels")
                self._resampled_labels_img_ = self._cache(
                    image.resample_img, func_memory_level=2)(
                        self.labels_img_, interpolation="nearest",
                        target_shape=imgs_.shape[:3],
                        target_affine=imgs_.affine)
            # Remove imgs_ from memory before loading the same image
            # in filter_and_extract.
            del imgs_

        target_shape = None
        target_affine = None
        if self.resampling_target == 'labels':
            target_shape = self._resampled_labels_img_.shape[:3]
            target_affine = self._resampled_labels_img_.affine

        params = get_params(NiftiLabelsMasker, self,
                            ignore=['resampling_target'])
        params['target_shape'] = target_shape
        params['target_affine'] = target_affine

        region_signals, labels_ = self._cache(
                filter_and_extract,
                ignore=['verbose', 'memory', 'memory_level'])(
            # Images
            imgs, _ExtractionFunctor(self._resampled_labels_img_,
                                     self.background_label),
            # Pre-processing
            params,
            confounds=confounds,
            dtype=self.dtype,
            # Caching
            memory=self.memory,
            memory_level=self.memory_level,
            verbose=self.verbose)

        self.labels_ = labels_

        return region_signals,labels_

    def inverse_transform(self, signals):
        """Compute voxel signals from region signals
        Any mask given at initialization is taken into account.
        Parameters
        ----------
        signals (2D numpy.ndarray)
            Signal for each region.
            shape: (number of scans, number of regions)
        Returns
        -------
        voxel_signals (Nifti1Image)
            Signal for each voxel
            shape: (number of scans, number of voxels)
        """
        from ..regions import signal_extraction

        self._check_fitted()

        logger.log("computing image from signals", verbose=self.verbose)
        return signal_extraction.signals_to_img_labels(
            signals, self._resampled_labels_img_, self.mask_img_,
            background_label=self.background_label)



class Extractor(joblib.Parallel):
    def __init__(self, data_dir_p, n_jobs=4,data_dir=""):
        super(Extractor, self).__init__()
        self.data_dir_p = data_dir_p
        self.n_jobs = n_jobs
        self.data_dir = data_dir


    # @classmethod
    def extract_atlasdata(self, df_withID, data_dir_p, n_jobs=4):
        ids = df_withID.SUBJECTKEY.values.tolist()
        a = ["ID"]
        num_cores2 = n_jobs
        df1 = pd.DataFrame(index=a.extend(list(range(len(ids)))), columns=range(135))
        df2 = pd.DataFrame(index=a.extend(list(range(len(ids)))), columns=range(135))                       
        
        # df1 = df1.append(Parallel(n_jobs=num_cores2)(delayed(self.extract_each1)(i, subjectID) for i,subjectID in enumerate(ids))).copy()
        for i,subjectID in enumerate(ids):
            df1 = df1.append(self.extract_each1(i, subjectID))
            df2 = df2.append(self.extract_each2(i, subjectID))

        return df1, df2

    # @staticmethod
    def extract_each1(self,i, subjectID):
        data_dir = self.data_dir
        data_dir_p = self.data_dir_p    
        id_name = str(subjectID)
        print("extract 1 : ",id_name)
        brain_loc = [glob.glob(os.path.join(data_dir,data_dir_p, "submission_*"+str(subjectID) + "*_brain.nii.gz"))[0]]
        gm_parc_loc = [glob.glob(os.path.join(data_dir,data_dir_p, "submission_*"+str(subjectID) + "*_gm_parc.nii.gz"))[0]]

        data = []
        brain = load_img(brain_loc)
        mask =  load_img(gm_parc_loc)
        data.append(brain)
        mask_data = (pd.Series(mask.get_data().flatten()).value_counts()).to_frame()
        mask_data.index = mask_data.index.astype('int64')
        mask_dataT = mask_data.T
        mask_dataT.index = [id_name]

        return mask_dataT

    # @staticmethod
    def extract_each2(self,i, subjectID):
        data_dir = self.data_dir
        data_dir_p = self.data_dir_p 
        id_name = str(subjectID)
        brain_loc = [glob.glob(os.path.join(data_dir,data_dir_p, "submission_*"+str(subjectID) + "*_brain.nii.gz"))[0]]
        gm_parc_loc = [glob.glob(os.path.join(data_dir,data_dir_p, "submission_*"+str(subjectID) + "*_gm_parc.nii.gz"))[0]]

        data = []
        brain = load_img(brain_loc)
        mask =  load_img(gm_parc_loc)
        data.append(brain)

        masker = NiftiLabelsMasker2(labels_img=mask, standardize=False,memory='nilearn_cache')
        masked_data,labels = masker.transform_single_imgs(imgs=data)#transform_single_imgs

        region_intensity = pd.DataFrame({'labels':labels, 'instensity':masked_data[0]})
        region_intensity = region_intensity.set_index('labels')
        region_intensity.index = region_intensity.index.astype('int64')
        region_intensityT = region_intensity.T
        region_intensityT.index=[id_name]

        return region_intensityT