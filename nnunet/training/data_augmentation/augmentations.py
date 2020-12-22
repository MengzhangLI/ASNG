import numpy as np
import copy
from batchgenerators.transforms import Compose, RenameTransform, GammaTransform, SpatialTransform
from batchgenerators.transforms import DataChannelSelectionTransform, SegChannelSelectionTransform
from batchgenerators.transforms import MirrorTransform, NumpyToTensor
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform
from nnunet.training.data_augmentation.pyramid_augmentations import MoveSegAsOneHotToData, \
    RemoveRandomConnectedComponentFromOneHotEncodingTransform, ApplyRandomBinaryOperatorTransform
from nnunet.training.data_augmentation.custom_transforms import Convert3DTo2DTransform, Convert2DTo3DTransform, \
    MaskTransform

default_3D_augmentation_params = {
    "selected_data_channels": None,
    "selected_seg_channels": None,
    "do_elastic": True, #
    "elastic_deform_alpha": (0., 900.),
    "elastic_deform_sigma": (9., 13.),
    "do_scaling": True, #
    "scale_range": (0.85, 1.25),
    "do_rotation": True, #
    "rotation_x": (-15./360 * 2. * np.pi, 15./360 * 2. * np.pi),
    "rotation_y": (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
    "rotation_z": (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
    "random_crop": False,
    "random_crop_dist_to_border": None,
    "do_gamma": True, #
    "gamma_retain_stats": True,
    "gamma_range": (0.7, 1.5),
    "p_gamma": 0.3,
    "num_threads": 12,
    "num_cached_per_thread": 1,
    "mirror": True,
    "mirror_axes": (0, 1, 2),
    "p_eldef": 0.2,
    "p_scale": 0.2,
    "p_rot": 0.2,
    "dummy_2D": False,
    "mask_was_used_for_normalization": False,
    "all_segmentation_labels": None,  # used for pyramid
    "move_last_seg_chanel_to_data": False,  # used for pyramid
    "border_mode_data": "constant",
    "advanced_pyramid_augmentations": False  # used for pyramid
}

def augment_list():
    l = [
        ("elastic_deform_alpha", 0., 450., 450., 900.),
        ("elastic_deform_sigma", 0., 7., 7., 14.),
        ("scale_range", 0.5, 1.0, 1.0, 1.5),
        ("rotation_x", -15./360 * 4. * np.pi, 0., 0., 15./360 * 4. * np.pi),
        ("rotation_y", -15./360 * 4. * np.pi, 0., 0., 15./360 * 4. * np.pi),
        ("rotation_z", -15./360 * 4. * np.pi, 0., 0., 15./360 * 4. * np.pi),
        ("gamma_range", 0.5, 1.0, 1.0, 1.5),
        ("p_eldef", 0., 1., None, None),
        ("p_scale", 0., 1., None, None),
        ("p_rot", 0., 1., None, None),
        ("p_gamma", 0., 1., None, None)
    ]
    return l

def Transforms(patch_size, params=default_3D_augmentation_params, border_val_seg=-1):
    tr_transforms = []
    if params.get("selected_data_channels") is not None:
        tr_transforms.append(DataChannelSelectionTransform(params.get("selected_data_channels"), data_key="data"))

    if params.get("selected_seg_channels") is not None:
        tr_transforms.append(SegChannelSelectionTransform(params.get("selected_seg_channels")))

    # don't do color augmentations while in 2d mode with 3d data because the color channel is overloaded!!
    if params.get("dummy_2D") is not None and params.get("dummy_2D"):
        tr_transforms.append(Convert3DTo2DTransform())
    tr_transforms.append(SpatialTransform(
        patch_size, patch_center_dist_from_border=None, do_elastic_deform=params.get("do_elastic"),
        alpha=params.get("elastic_deform_alpha"), sigma=params.get("elastic_deform_sigma"),
        do_rotation=params.get("do_rotation"), angle_x=params.get("rotation_x"), angle_y=params.get("rotation_y"),
        angle_z=params.get("rotation_z"), do_scale=params.get("do_scaling"), scale=params.get("scale_range"),
        border_mode_data=params.get("border_mode_data"), border_cval_data=0, order_data=3, border_mode_seg="constant",
        border_cval_seg=border_val_seg,
        order_seg=1, random_crop=params.get("random_crop"), p_el_per_sample=params.get("p_eldef"),
        p_scale_per_sample=params.get("p_scale"), p_rot_per_sample=params.get("p_rot")
    ))
    if params.get("dummy_2D") is not None and params.get("dummy_2D"):
        tr_transforms.append(Convert2DTo3DTransform())

    if params.get("do_gamma"):
        tr_transforms.append(GammaTransform(params.get("gamma_range"), False, True, retain_stats=params.get("gamma_retain_stats"), p_per_sample=params["p_gamma"]))

    tr_transforms.append(MirrorTransform(params.get("mirror_axes")))
    if params.get("mask_was_used_for_normalization") is not None:
        mask_was_used_for_normalization = params.get("mask_was_used_for_normalization")
        tr_transforms.append(MaskTransform(mask_was_used_for_normalization, mask_idx_in_seg=0, set_outside_to=0))

    tr_transforms.append(RemoveLabelTransform(-1, 0))

    if params.get("move_last_seg_chanel_to_data") is not None and params.get("move_last_seg_chanel_to_data"):
        tr_transforms.append(MoveSegAsOneHotToData(1, params.get("all_segmentation_labels"), 'seg', 'data'))
        if params.get("advanced_pyramid_augmentations") and not None and params.get("advanced_pyramid_augmentations"):
            tr_transforms.append(ApplyRandomBinaryOperatorTransform(channel_idx=list(range(-len(params.get("all_segmentation_labels")), 0)),
                                                                    p_per_sample=0.4,
                                                                    key="data",
                                                                    strel_size=(1, 8)))
            tr_transforms.append(RemoveRandomConnectedComponentFromOneHotEncodingTransform(channel_idx=list(range(-len(params.get("all_segmentation_labels")), 0)),
                                                                                           key="data",
                                                                                           p_per_sample=0.2,
                                                                                           fill_with_other_class_p=0.0,
                                                                                           dont_do_if_covers_more_than_X_percent=0.15))

    tr_transforms.append(RenameTransform('seg', 'target', True))
    tr_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
    tr_transforms = Compose(tr_transforms)
    return tr_transforms

class AutoAugment:
    def __init__(self, patch_size, params=default_3D_augmentation_params, N=11):
        self.params = params
        self.augment_list = augment_list()
        self.patch_size = patch_size
        self.N = N

    def __call__(self, data, masks):
        #assert len(self.augment_list) == len(masks)
        degrees = []
        for i in range(len(masks)):
            degrees.append(np.argmax(masks[i]))
        LR = 0
        for value in self.augment_list:
            if len(value)>3:
                LR += 1
        LR_degrees = []
        begin = 0
        for i in range(LR):
            LR_degrees.append((degrees[begin], degrees[begin+1]))
            begin += 2
        while begin<len(degrees):
            LR_degrees.append((degrees[begin],))
            begin += 1
        for index,(name, left_low, left_high, right_low, right_high) in enumerate(self.augment_list):
            curDegrees = LR_degrees[index]
            if right_low is not None:
                left_degree = (left_high - left_low)/(self.N-1)*curDegrees[0] + left_low
                right_degree = (right_high - right_low)/(self.N-1)*curDegrees[1] + right_low
                self.params[name] = (left_degree, right_degree)
            else:
                degree = (left_high - left_low)/(self.N-1)*curDegrees[0] + left_low
                self.params[name] = degree
        transforms = Transforms(self.patch_size, self.params)
        return transforms(**data)


class CategoricalASNG:
    """Adaptive stochastic natural gradient method on multivariate categorical distribution.
    Args:
        categories (numpy.ndarray): Array containing the numbers of categories of each dimension.
        alpha (float): Threshold of SNR in ASNG algorithm.
        init_delta (float): Initial value of delta.
        Delta_max (float): Maximum value of Delta.
        init_theta (numpy.ndarray): Initial parameter of theta. Its shape must be (len(categories), max(categories)).
    """

    def __init__(self, categories, alpha=1.5, init_delta=1., Delta_max=np.inf, init_theta=None):

        self.p_model = Categorical(categories)

        if init_theta is not None:
            self.p_model.theta = init_theta

        self.N = np.sum(categories - 1)
        self.delta = init_delta
        self.Delta = 1.
        self.Delta_max = np.inf
        self.alpha = alpha
        self.gamma = 0.
        self.s = np.zeros(self.N)

    def get_delta(self):
        return self.delta/self.Delta

    def sampling(self):
        return self.p_model.sampling()

    def update(self, Ms, losses, range_restriction=True):
        delta = self.get_delta() #学习率
        beta = delta * self.N**-0.5

        u, idx = self.utility(losses)
        mu_W, var_W = u.mean(), u.var()
        if var_W == 0:
            return

        ngrad = np.mean((u - mu_W)[:, np.newaxis, np.newaxis] * (Ms[idx] - self.p_model.theta), axis=0) #计算的关于\theta的梯度。

        # Too small natural gradient leads ngnorm to 0.
        if (np.abs(ngrad) < 1e-18).all():
            print('skip update')
            return

        s_latter = []
        for i, K in enumerate(self.p_model.C):
            theta_i = self.p_model.theta[i, :K - 1]
            theta_K = self.p_model.theta[i, K - 1]
            s_i = 1/np.sqrt(theta_i) * ngrad[i, :K - 1]
            s_i += np.sqrt(theta_i) * ngrad[i, :K - 1].sum() / (theta_K + np.sqrt(theta_K))
            s_latter += list(s_i)
        s_latter = np.array(s_latter)

        ngnorm = np.sqrt(np.sum(s_latter**2)) #把梯度normalize
        dp = ngrad/ngnorm
        assert not np.isnan(dp).any(), (ngrad, ngnorm)
        self.p_model.theta += delta * dp #更新参数
        #确保参数和为1
        for i in range(self.p_model.d):
            ci = self.p_model.C[i]

            # Constraint for theta (minimum value of theta and sum of theta = 1.0)
            theta_min = 1. / (self.p_model.valid_d * (ci - 1)) if range_restriction and ci > 1 else 0.
            self.p_model.theta[i, :ci] = np.maximum(self.p_model.theta[i, :ci], theta_min)
            theta_sum = self.p_model.theta[i, :ci].sum()
            tmp = theta_sum - theta_min * ci
            self.p_model.theta[i, :ci] -= (theta_sum - 1.) * (self.p_model.theta[i, :ci] - theta_min) / tmp

            # Ensure the summation to 1
            self.p_model.theta[i, :ci] /= self.p_model.theta[i, :ci].sum()

        self.s = (1 - beta)*self.s + np.sqrt(beta*(2 - beta))*s_latter/ngnorm
        self.gamma = (1 - beta)**2 * self.gamma + beta*(2 - beta)
        self.Delta *= np.exp(beta * (self.gamma - np.dot(self.s, self.s)/self.alpha))
        self.Delta = np.minimum(self.Delta, self.Delta_max)


    @staticmethod
    def utility(f, rho=0.25, negative=True):
        """
        Ranking Based Utility Transformation
        w(f(x)) / lambda =
            1/mu  if rank(x) <= mu
            0     if mu < rank(x) < lambda - mu
            -1/mu if lambda - mu <= rank(x)
        where rank(x) is the number of at least equally good
        points, including it self.
        The number of good and bad points, mu, is ceil(lambda/4).
        That is,
            mu = 1 if lambda = 2
            mu = 1 if lambda = 4
            mu = 2 if lambda = 6, etc.
        If there exist tie points, the utility values are
        equally distributed for these points.
        """
        eps = 1e-14
        idx = np.argsort(f)
        lam = len(f)
        mu = int(np.ceil(lam * rho))
        _w = np.zeros(lam)
        _w[:mu] = 1 / mu
        _w[lam - mu:] = -1 / mu if negative else 0
        w = np.zeros(lam)
        istart = 0
        for i in range(f.shape[0] - 1):
            if f[idx[i + 1]] - f[idx[i]] < eps * f[idx[i]]:
                pass
            elif istart < i:
                w[istart:i + 1] = np.mean(_w[istart:i + 1])
                istart = i + 1
            else:
                w[i] = _w[i]
                istart = i + 1
        w[istart:] = np.mean(_w[istart:])
        return w, idx

class Categorical(object):
    """
    Categorical distribution for categorical variables parametrized by :math:`\\{ \\theta \\}_{i=1}^{(d \\times K)}`.
    :param categories: the numbers of categories
    :type categories: array_like, shape(d), dtype=int
    """
    def __init__(self, categories):
        self.d = len(categories)
        self.C = categories
        self.Cmax = np.max(categories)
        self.theta = np.zeros((self.d, self.Cmax))
        # initialize theta by 1/C for each dimensions
        for i in range(self.d):
            self.theta[i, :self.C[i]] = 1./self.C[i]
        # pad zeros to unused elements
        for i in range(self.d):
            self.theta[i, self.C[i]:] = 0.
        # number of valid parameters
        self.valid_param_num = int(np.sum(self.C - 1))
        # valid dimension size
        self.valid_d = len(self.C[self.C > 1])
        self.opsNum = len(augment_list())

    def sampling(self):
        """
        Draw a sample from the categorical distribution.
        :return: sampled variables from the categorical distribution (one-hot representation)
        :rtype: array_like, shape=(d, Cmax), dtype=bool
        """
        rand = np.random.rand(self.d, 1)    # range of random number is [0, 1)
        cum_theta = self.theta.cumsum(axis=1)    # (d, Cmax)

        # x[i, j] becomes 1 iff cum_theta[i, j] - theta[i, j] <= rand[i] < cum_theta[i, j]
        x = (cum_theta - self.theta <= rand) & (rand < cum_theta)
        x = x+0 #转True为1，False为0.
        x = x.astype("float")
        return x

    def mle(self):
        """
        Return the most likely categories.
        :return: categorical variables (one-hot representation)
        :rtype: array_like, shape=(d, Cmax), dtype=bool
        """
        m = self.theta.argmax(axis=1)
        x = np.zeros((self.d, self.Cmax))
        for i, c in enumerate(m):
            x[i, c] = 1
        return x


