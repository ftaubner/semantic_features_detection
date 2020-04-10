#cython: language_level=3

cimport numpy as np
import numpy as np
ctypedef np.uint8_t uint8

cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def get_binary_masks(instance_mask, unsigned short[:] instance_ids, unsigned short[:,:] instance_im):
    cdef uint8[:,:,:] instance_mask_view = instance_mask

    for i in xrange(instance_im.shape[0]):
        for j in xrange(instance_im.shape[1]):
            for l in xrange(instance_ids.shape[0]):
                if instance_ids[l] == instance_im[i, j]:
                    instance_mask_view[i, j, l] = True
                    break
                    
    return instance_mask
