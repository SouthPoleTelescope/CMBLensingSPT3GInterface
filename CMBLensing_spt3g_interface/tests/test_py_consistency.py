from numpy.testing import assert_allclose

print("Loading CMBLensing...")
from CMBLensing_spt3g_interface.julia_jlcommand import jl
from CMBLensing_spt3g_interface.CMBLensing_spt3g_interface import *

from spt3g.mapspectra.basicmaputils import map_to_ft
from spt3g.lensing.map_spec_utils import calculate_teb


# load field
print("Loading dataset...")
f = jl("load_sim_dataset(θpix=2, Nside=32, pol=:IP)").f;

# make sure that Fourier conventions are consistent, i.e.
# that loading a field in Fourier space and FFTing to map space
# is the same as loading the field directly in map space (and vice versa).
for k in jl("jl_keys").keys():
    flatsky_map  = toFlatSkyMap(jl("$f[jl_keys[$k]]"))
    map_spectrum = toMapSpectrum2D(jl("$f[jl_keys[$k]]"))

    assert_allclose(flatsky_map, map_spectrum.get_rmap(), rtol=1e-12, atol=1e-4,
        err_msg="{} loaded in Fourier-space and converted to map-space is not the same as {} loaded in map-space".format(k,k))

    assert_allclose(map_spectrum, map_to_ft(flatsky_map).get_real(), rtol=1e-12, atol=1e-4,
        err_msg="{} loaded in map-space and converted to Fourier-space is not the same as {} loaded in Fourier-space".format(k,k))


# make sure sure that toFrame and toMapSpectraTEB work,
# and that the two packages' Q/U -> E/B conventions are consistent.
tqu = toFrame(f, keys="TQU", constructor=toFlatSkyMap)
teb_3g = calculate_teb(tqu)
teb_cmbl = toMapSpectraTEB(f)
for k in "TEB":
    assert_allclose(teb_3g[k], teb_cmbl[k], rtol=1e-12, atol=1e-4,
        err_msg="{} made from map-space TQU with calculate_teb is not the same as {} loaded in Fourier space".format(k,k))
