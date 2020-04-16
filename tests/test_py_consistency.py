from CMBLensing_spt3g_interface.CMBLensing_spt3g_interface import *
from CMBLensing_spt3g_interface.julia_jlcommand import jl

from spt3g.mapspectra.basicmaputils import map_to_ft
from spt3g.lensing.map_spec_utils import calculate_teb
from spt3g.core import G3Units
uK = G3Units.uK

# load field
f = jl("load_sim_dataset(θpix=2, Nside=512, pol=:IP, rfid=1e-8)").f;

# make sure that Fourier conventions are consistent, i.e.
# that loading a field in Fourier space and FFTing to map space
# is the same as loading the field directly in map space (and vice versa). 
for k in jl("jl_keys").keys():
    flatsky_map  = toFlatSkyMap(jl("$f[jl_keys[$k]]")) * uK
    map_spectrum = toMapSpectrum2D(jl("$f[jl_keys[$k]]")) * uK
    
    assert np.allclose(flatsky_map, map_spectrum.get_rmap(), 
                       rtol=1e-16, atol=1e-4), \
        "{} loaded in Fourier-space and converted to map-space is not the same as {} loaded in map-space".format(k,k)
    
    assert np.allclose(map_spectrum, map_to_ft(flatsky_map).get_real(), 
                       rtol=1e-16, atol=1e-8), \
        "{} loaded in map-space and converted to Fourier-space is not the same as {} loaded in Fourier-space".format(k,k)
    

# make sure sure that toFrame and toMapSpectraTEB work,
# and that the two packages' Q/U -> E/B conventions are consistent. 
tqu = toFrame(f, keys="TQU", constructor=toFlatSkyMap)
teb_3g = calculate_teb(tqu)
teb_cmbl = toMapSpectraTEB(f)
for k in "TEB":
    assert np.allclose(teb_3g[k], teb_cmbl[k], rtol=1e-16, atol=1e-7), \
            "{} made from map-space TQU with calculate_teb is not the same as {} loaded in Fourier space".format(k,k)