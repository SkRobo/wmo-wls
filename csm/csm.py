from cffi import FFI
import numpy as np
import os, sys
import csm # self import to get path to the bundled files

class CSMResult:
    is_valid = True     #: Is result of matching valid
    x = [0, 0, 0]       #: Scan matching result (x,y,theta)
    iterations = 0      #: Number of iterations done
    nvalid = 0          #: Number of valid correspondence in the end
    error = 0           #: Total correspondence error
    cov_x_m = None

    def __init__(self, pointer):
        '''Creates result object from the CFFI pointer

        Parameters
        ----------
        pointer : <cdata 'struct sm_result *' owning 48 bytes>
            CFFI pointer to `sm_result` structure
        '''
        self.is_valid = pointer.valid == 1
        self.x = np.array([pointer.x[i] for i in range(3)], np.float64)
        self.iterations = pointer.iterations
        self.nvalid = pointer.nvalid
        self.error = pointer.error
        self.p = pointer

class CSM:
    _ffi = None
    _lib = None

    scan_size = 1040
    #: Size of matched scans
    scan_fov = 260
    #: Field of view of scanner (deg)

    max_angular_correction_deg          = 90
    #: Maximum angular displacement between scans (deg)
    max_linear_correction               = 2
    #: Maximum translation between scans (m)
    max_iterations                      = 1000
    #: When to stop
    epsilon_xy                          = 0.0001
    #: Translation threshold for stopping
    epsilon_theta                       = 0.0001
    #: Rotation threshold for stopping
    max_correspondence_dist             = 2
    #: Maximum distance for a correspondence to be valid
    sigma                               = 0.01
    #: Noise in the scan
    use_corr_tricks                     = 1
    #: Use smart tricks for finding correspondences.
    #: Only influences speed; not convergence.
    restart                             = 1
    #: Restart if error under threshold (0 or 1)
    restart_threshold_mean_error        = 0.01
    #: Threshold for restarting
    restart_dt                          = 0.01
    #: Displacement for restarting
    restart_dtheta                      = 0.0261799
    #: Displacement for restarting
    clustering_threshold                = 0.05
    #: Max distance threshold for clustering
    orientation_neighbourhood           = 3
    #: Number of neighbour rays used to estimate the orientation.
    use_point_to_line_distance          = 1
    #: If 1, use PlICP; if 0, use vanilla ICP.
    do_alpha_test                       = 0
    #: Discard correspondences based on the angles
    do_alpha_test_thresholdDeg          = 20
    #: Threshold for alpha test
    outliers_maxPerc                    = 0.95
    #: Percentage of correspondences to consider: if 0.9,
    #: always discard the top 10% of correspondences with more error
    outliers_adaptive_order             = 0.7
    #: Parameters describing a simple adaptive algorithm for discarding.
    #: 1) Order the errors.
    #: 2) Choose the percentile according to ``outliers_adaptive_order``.
    #: (if it is 0.7, get the 70% percentile)
    #: 3) Define an adaptive threshold multiplying ``outliers_adaptive_mult``
    #: with the value of the error at the chosen percentile.
    #: 4) Discard correspondences over the threshold.
    #: This is useful to be conservative; yet remove the biggest errors.
    outliers_adaptive_mult              = 2
    #: Look description for ``outliers_adaptive_mult``
    outliers_remove_doubles             = 1
    #: Do not allow two different correspondences to share a point 
    do_visibility_test                  = 0
    #: I believe this trick is documented in one of the papers by Guttman
    #: (but I can't find the reference). Or perhaps I was told by him directly. 
    #: If you already have a guess of the solution, you can compute the polar
    #: angle of the points of one scan in the new position. If the polar angle
    #: is not a monotone function of the readings index, it means that the
    #: surface is not visible in the next position. If it is not visible, then
    #: we don't use it for matching. This is confusing without a picture!
    #: To understand what's going on, make a drawing in which a surface is not
    #: visible in one of the poses.
    do_compute_covariance               = 0
    #: Use the method in http://purl.org/censi/2006/icpcov to compute
    #: the matching covariance.
    debug_verify_tricks                 = 0
    #: Checks that find_correspondences_tricks give the right answer
    laser                               = [0, 0, 0]
    #: Pose of sensor with respect to robot: used for computing
    #: the first estimate given the odometry.
    min_reading                         = 0.02
    #: Maximum distance to use
    max_reading                         = 1000
    #: Minimum distance to use
    use_ml_weights                      = 0
    #: If 1, the field "true_alpha" is used to compute the incidence
    #: beta, and the factor (1/cos^2(beta)) used to weight the impact
    #: of each correspondence. This works fabolously if doing localization,
    #: that is the first scan has no noise.
    #: If "true_alpha" is not available, it uses "alpha".
    use_sigma_weights                   = 0
    #: If 1, the field "readings_sigma" is used to weight the correspondence
    #: by 1/sigma^2


    _param_keys = [
        'max_angular_correction_deg',
        'max_linear_correction',
        'max_iterations',
        'epsilon_xy',
        'epsilon_theta',
        'max_correspondence_dist',
        'sigma',
        'use_corr_tricks',
        'restart',
        'restart_threshold_mean_error',
        'restart_dt',
        'restart_dtheta',
        'clustering_threshold',
        'orientation_neighbourhood',
        'use_point_to_line_distance',
        'do_alpha_test',
        'do_alpha_test_thresholdDeg',
        'outliers_maxPerc',
        'outliers_adaptive_order',
        'outliers_adaptive_mult',
        'do_visibility_test',
        'outliers_remove_doubles',
        'do_compute_covariance',
        'debug_verify_tricks',
        'min_reading',
        'max_reading',
        'use_ml_weights',
        'use_sigma_weights',
    ]

    def __init__(self, scan_size=1040, scan_fov=260, params=None):
        '''Creates CSM matcher object. Do not forget to set `scan_size` and
        `scan_fov` according to your sensor model. Currently only scans with
        fixed size are supported.

        Parameters
        ----------
        scan_size : int
            Number of measurments in scans for matching
        scan_fov : int
            Field of view of sensor which provids scans
        params : dict, optional
            Matching algorithm parameters. For list of available parameters and
            their description refer to class attributes description.
        '''
        self.scan_size = scan_size
        self.scan_fov = scan_fov

        self._ffi = FFI()
        import csm
        loc_dir = csm.__path__[0]
        with open(os.path.join(loc_dir, 'headers.h')) as f:
            self._ffi.cdef(f.read())
        self._lib = self._ffi.dlopen(os.path.join(loc_dir, 'libcsm.so'))

        self._sm_params = self._ffi.new('struct sm_params *')
        self._sm_result = self._ffi.new('struct sm_result *')

        self.update_params(params)

    @staticmethod
    def _set3(ptr, data):
        '''Sets values for 3D vector in the form of CFFI pointer'''
        for i in range(3):
            ptr[i] = data[i]

    def update_params(self, params=None):
        '''Updates matching algorithm parameters. Either pass new arguments in
        `params` or update according attribute and then call this function.

        Parameters
        ----------
        params : dict, optional
            Parameters to update
        '''
        if params is None:
            params = {}
        for key in params.keys():
            if key not in self._param_keys and key != 'laser':
                raise ValueError('Unknown parameter: %s' % key)
        for key in self._param_keys:
            if key in params:
                self.__setattr__(key, params[key])
            self._sm_params.__setattr__(key, self.__getattribute__(key))
        if 'laser' in params:
            if np.array(params['laser']).shape != (3,):
                raise ValueError('Wrong laser parameter')
            self.laser = params['laser']
        self._set3(self._sm_params.laser, self.laser)

    def _get_ldp(self, scan, odometry=None):
        '''Constructs LDP object from scan and returns CFFI pointer'''
        cscan = scan.astype(np.float64)
        valid = np.logical_and(cscan>self.min_reading, cscan<self.max_reading)
        cscan[~valid] = np.nan
        cvalid = valid.astype(np.intc)

        ldp = self._lib.ld_alloc_new(self.scan_size)
        theta = np.radians(self.scan_fov/2, dtype=np.float64)
        ldp.min_theta = -theta
        ldp.max_theta = theta

        thetas = np.linspace(-theta, theta, self.scan_size, dtype=np.float64)
        self._ffi.memmove(ldp.theta,
            self._ffi.cast('double *', thetas.ctypes.data), thetas.nbytes)
        self._ffi.memmove(ldp.readings,
            self._ffi.cast('double *', cscan.ctypes.data), cscan.nbytes)
        self._ffi.memmove(ldp.valid,
            self._ffi.cast('int *', cvalid.ctypes.data), cvalid.nbytes)
        if odometry is not None:
            self._set3(ldp.odometry, odometry)
        return ldp

    def match(self, ref_scan, scan, guess=None):
        '''Matches `ref_scan` and `scan` with parameters stored in the matcher.

        Parameters
        ----------
        ref_scan : list, array
            Reference (first) scan. (m)
        scan : list, array
            Second scan. (m)
        guess: [x, y, theta]
            Initiall position guess

        Returns
        -------
        CSMResult
            Matching result
        '''
        if len(ref_scan) != self.scan_size:
            raise ValueError('Wrong size of ref_scan: %d vs %d' %
                                (len(ref_scan), self.scan_size))
        if len(scan) != self.scan_size:
            raise ValueError('Wrong size of scan: %d vs %d' %
                                (len(scan), self.scan_size))
        guess = [0, 0, 0] if guess is None else guess
        if len(guess) != 3:
            raise ValueError('Wrong size of guess: %d vs 3' % len(ref_scan))
        self._sm_params.laser_ref = self._get_ldp(ref_scan)
        self._sm_params.laser_sens = self._get_ldp(scan)
        self._set3(self._sm_params.first_guess, guess)
        self._lib.sm_icp(self._sm_params, self._sm_result)
        self._lib.ld_free(self._sm_params.laser_ref)
        self._lib.ld_free(self._sm_params.laser_sens)
        return CSMResult(self._sm_result)
