#cython: language_level=3

cimport numpy as np
import numpy as np
ctypedef np.uint8_t uint8
ctypedef np.int32_t int32

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

                    
@cython.boundscheck(False)
@cython.wraparound(False)
def extract_bboxes(uint8[:,:,:] instance_mask):
    #y1, x1, y2, x2
    bboxes = -np.ones([instance_mask.shape[2], 4], dtype=np.int32)
    cdef int32[:,:] bboxes_view = bboxes

    for i_ in range(instance_mask.shape[0]):
        for j_ in range(instance_mask.shape[1]):
            for l in range(instance_mask.shape[2]):
                if instance_mask[i_, j_, l]:
                    if bboxes_view[l, 0] == -1 or i_ < bboxes_view[l, 0]:
                        bboxes_view[l, 0] = i_
                    if bboxes_view[l, 2] == -1 or i_ > bboxes_view[l, 2] - 1:
                        bboxes_view[l, 2] = i_ + 1
                    if bboxes_view[l, 1] == -1 or j_ < bboxes_view[l, 1]:
                        bboxes_view[l, 1] = j_
                    if bboxes_view[l, 3] == -1 or j_ > bboxes_view[l, 3] - 1:
                        bboxes_view[l, 3] = j_ + 1
                    break

    return bboxes