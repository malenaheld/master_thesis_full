import numpy as np
import pymaster as nmt
import healpy as hp
import pymaster.utils as ut


def get_fields_from_map(delta, mask, fls, nside, sub_mean=True, iter=3):
    """ Returns NaMaster fields from a given map.

    Args:
        delta: input map.
        mask: mask associated with this field. Note that we assume the
            mask to be binary.
        sub_mean: ensure that all filtered-squared maps have zero mean?
        iter: number of Jacobi iterations when computing SHTs.

    Returns: dictionary of NaMaster fields. 'fN' denotes the original
        overdensity field, while 'f1', 'f2', etc. correspond to the
        filtered-squared maps of the different filters.
    """
    flds = {}
    # Mask things for safety.
    # Note we're assuming mask to be binary at this stage, otherwise this is wrong.
    mp = delta*mask
    if sub_mean:
        mp = mp - mask * np.sum(mp*mask)/np.sum(mask)
    flds['fN'] = nmt.NmtField(mask, [mp], n_iter=iter)

    # Alm of the original map
    alm = hp.map2alm(mp, iter=iter)

    # Filtered-squared maps for each filter
    mp_filt_sq = np.array([hp.alm2map(hp.almxfl(alm, fl), nside)**2 for fl in fls])
    if sub_mean:
        mp_filt_sq = np.array([m-mask*np.sum(m*mask)/np.sum(mask)
                               for m in mp_filt_sq])
    for i, m in enumerate(mp_filt_sq):
        flds[f'f{i}'] = nmt.NmtField(mask, [m], n_iter=iter)

    return flds

def get_cl(f1, f2, wsp, decouple=True):
    """ Given two NaMaster fields and a workspace, compute the power spectrum.

    Args:
        f1, f2: NaMaster fields to correlate.
        wsp: associated workspace (pass `None` if not attempting to undo
            the mode-coupling).
        decouple: if True, the standard pseudo-Cl, accounting for mode-coupling,
            will be computed. If False, no mode-coupling will be accounted for.
            The naive power spectrum will just be divided by the sky fraction.

    Returns: measured power spectrum.
    """
    pcl = nmt.compute_coupled_cell(f1, f2)
    if decouple:
        cl = wsp.decouple_cell(pcl)
    else:
        m1 = f1.get_mask()
        m2 = f2.get_mask()
        cl = pcl/np.mean(m1*m2)
    return cl

def get_filters(nbands, lmax):
    dell = lmax // nbands

    filters = np.zeros((nbands, lmax))
    for i in range(nbands):
        filters[i, i*dell:(i+1)*dell] = 1.0
    return filters

def get_fields_from_map_wtemp(delta, mask, fls, nside, templates=None, sub_mean=True, iter=3):
    """ Returns NaMaster fields from a given map.

    Args:
        delta: input map.
        mask: mask associated with this field. Note that we assume the
            mask to be binary.
        sub_mean: ensure that all filtered-squared maps have zero mean?
        iter: number of Jacobi iterations when computing SHTs.

    Returns: dictionary of NaMaster fields. 'fN' denotes the original
        overdensity field, while 'f1', 'f2', etc. correspond to the
        filtered-squared maps of the different filters.
    """
    flds = {}
    # Mask things for safety.
    # Note we're assuming mask to be binary at this stage, otherwise this is wrong.
    mp = delta*mask
    if sub_mean:
        mp = mp - mask * np.sum(mp*mask)/np.sum(mask)
    flds['fN'] = nmt.NmtField(mask, [mp], templates=templates, n_iter=iter)

    mp = flds['fN'].get_maps()[0]
    # Alm of the original map
    alm = hp.map2alm(mp, iter=iter)

    # Filtered-squared maps for each filter
    mp_filt_sq = np.array([hp.alm2map(hp.almxfl(alm, fl), nside)**2 for fl in fls])
    if sub_mean:
        mp_filt_sq = np.array([m-mask*np.sum(m*mask)/np.sum(mask)
                               for m in mp_filt_sq])
    for i, m in enumerate(mp_filt_sq):
        flds[f'f{i}'] = nmt.NmtField(mask, [m], n_iter=iter)#, templates=[[templates[0][0]**2]])

    return flds



def get_fields_from_map_wtemp2(delta, mask, fls, nside, templates=None, sub_mean=True, iter=3):
    """ Returns NaMaster fields from a given map.

    Args:
        delta: input map.
        mask: mask associated with this field. Note that we assume the
            mask to be binary.
        sub_mean: ensure that all filtered-squared maps have zero mean?
        iter: number of Jacobi iterations when computing SHTs.

    Returns: dictionary of NaMaster fields. 'fN' denotes the original
        overdensity field, while 'f1', 'f2', etc. correspond to the
        filtered-squared maps of the different filters.
    """
    flds = {}
    # Mask things for safety.
    # Note we're assuming mask to be binary at this stage, otherwise this is wrong.

    mp = delta*mask
    if sub_mean:
        mp = mp - mask * np.sum(mp*mask)/np.sum(mask)
    flds['fN'] = nmt.NmtField(mask, [mp], templates=templates, n_iter=iter)
    # Alm of the original map
    alm = hp.map2alm(mp, iter=iter)

#    template = get_filtered_field(templates[0][0], mask, fls, nside, sub_mean=True, iter=3)

    # Filtered-squared maps for each filter
    mp_filt_sq = np.array([hp.alm2map(hp.almxfl(alm, fl), nside)**2 for fl in fls])
    if sub_mean:
        mp_filt_sq = np.array([m-mask*np.sum(m*mask)/np.sum(mask)
                               for m in mp_filt_sq])
    for i, m in enumerate(mp_filt_sq):
        flds[f'f{i}'] = nmt.NmtField(mask, [m], n_iter=iter, templates=[[templates[0][0]**2]])

    return flds

def get_fields_from_map_wtemp3(delta, mask, fls, nside, templates=None, sub_mean=True, iter=3):
    """ Returns NaMaster fields from a given map.

    Args:
        delta: input map.
        mask: mask associated with this field. Note that we assume the
            mask to be binary.
        sub_mean: ensure that all filtered-squared maps have zero mean?
        iter: number of Jacobi iterations when computing SHTs.

    Returns: dictionary of NaMaster fields. 'fN' denotes the original
        overdensity field, while 'f1', 'f2', etc. correspond to the
        filtered-squared maps of the different filters.
    """
    flds = {}
    # Mask things for safety.
    # Note we're assuming mask to be binary at this stage, otherwise this is wrong.
    mp = delta*mask
    if sub_mean:
        mp = mp - mask * np.sum(mp*mask)/np.sum(mask)
    flds['fN'] = nmt.NmtField(mask, [mp], templates=templates, n_iter=iter)
    # Alm of the original map
    alm = hp.map2alm(mp, iter=iter)

    # Filtered-squared maps for each filter
    mp_filt_sq = np.array([hp.alm2map(hp.almxfl(alm, fl), nside)**2 for fl in fls])
    if sub_mean:
        mp_filt_sq = np.array([m-mask*np.sum(m*mask)/np.sum(mask)
                               for m in mp_filt_sq])
    for i, m in enumerate(mp_filt_sq):
        flds[f'f{i}'] = nmt.NmtField(mask, [m], n_iter=iter, templates=templates)

    return flds

def get_filtered_field(delta, mask, fls, nside, sub_mean=True, iter=3):
    """ Returns NaMaster fields from a given map.

    Args:
        delta: input map.
        mask: mask associated with this field. Note that we assume the
            mask to be binary.
        sub_mean: ensure that all filtered-squared maps have zero mean?
        iter: number of Jacobi iterations when computing SHTs.

    Returns: dictionary of NaMaster fields. 'fN' denotes the original
        overdensity field, while 'f1', 'f2', etc. correspond to the
        filtered-squared maps of the different filters.
    """
    flds = {}
    # Mask things for safety.
    # Note we're assuming mask to be binary at this stage, otherwise this is wrong.
    mp = delta*mask
    if sub_mean:
        mp = mp - mask * np.sum(mp*mask)/np.sum(mask)
    flds['fN'] = nmt.NmtField(mask, [mp], n_iter=iter)

    # Alm of the original map
    alm = hp.map2alm(mp, iter=iter)

    # Filtered-squared maps for each filter
    mp_filt_sq = np.array([hp.alm2map(hp.almxfl(alm, fl), nside) for fl in fls])
    if sub_mean:
        mp_filt_sq = np.array([m-mask*np.sum(m*mask)/np.sum(mask)
                               for m in mp_filt_sq])
    for i, m in enumerate(mp_filt_sq):
        flds[f'f{i}'] = nmt.NmtField(mask, [m], n_iter=iter)

    return flds

def get_fields_from_map_wtemp_filt(delta, mask, fls, nside, templates=None, sub_mean=True, iter=3):
    """ Returns NaMaster fields from a given map.

    Args:
        delta: input map.
        mask: mask associated with this field. Note that we assume the
            mask to be binary.
        sub_mean: ensure that all filtered-squared maps have zero mean?
        iter: number of Jacobi iterations when computing SHTs.

    Returns: dictionary of NaMaster fields. 'fN' denotes the original
        overdensity field, while 'f1', 'f2', etc. correspond to the
        filtered-squared maps of the different filters.
    """
    flds = {}
    # Mask things for safety.
    # Note we're assuming mask to be binary at this stage, otherwise this is wrong.

    mp = delta*mask
    if sub_mean:
        mp = mp - mask * np.sum(mp*mask)/np.sum(mask)
    flds['fN'] = nmt.NmtField(mask, [mp], templates=templates, n_iter=iter)
    # Alm of the original map
    alm = hp.map2alm(mp, iter=iter)

    template = get_filtered_field(templates[0][0], mask, fls, nside, sub_mean=True, iter=3)

    # Filtered-squared maps for each filter
    mp_filt_sq = np.array([hp.alm2map(hp.almxfl(alm, fl), nside)**2 for fl in fls])
    if sub_mean:
        mp_filt_sq = np.array([m-mask*np.sum(m*mask)/np.sum(mask)
                               for m in mp_filt_sq])
    for i, m in enumerate(mp_filt_sq):
        flds[f'f{i}'] = nmt.NmtField(mask, [m], n_iter=iter, templates=[[template[f'f{i}'].get_maps()[0]**2]])

    return flds


def first_term(fld1, fld2, fsbg, n_iter=None):
    if n_iter is None:
        n_iter = ut.nmt_params.n_iter_default

    pcl_1t = np.zeros((fld1.n_temp, fld2.n_temp, fld1.nmaps, fld2.nmaps, fld1.ainfo.lmax+1))
    for ij, tj in enumerate(fld1.temp):
        ## SHT(g_i)
        fsbb = ut.map2alm(tj, fld1.spin, fld1.minfo, fld1.ainfo, n_iter=n_iter)   ## transform the template (map) into spherical harmonic coefficients (alm)

        ## SHT(g_i)*fsb
        fsbb = np.array([
            np.sum([hp.almxfl(fsbb[m], fsbg[m,n],
                            mmax=fld1.ainfo.mmax) 
                            for m in range(fld1.nmaps)], axis=0) 
                            for n in range(fld2.nmaps)])
        

        ## PCL(SHT^-1(SHT(g_i)*fsb), gj)
        for jj, g_j in enumerate(fld1.alm_temp):
            clij = np.array([[hp.alm2cl(a1, a2, lmax=fld1.ainfo.lmax)
                            for a2 in fsbb]
                            for a1 in g_j])
            
            pcl_1t[jj, ij, :, :, :] = clij

    t1 = np.zeros((fld1.nmaps, fld2.nmaps, fld1.ainfo.lmax+1))
    t1 = - np.einsum('ij, ijklm', fld1.iM, pcl_1t)
    return t1
