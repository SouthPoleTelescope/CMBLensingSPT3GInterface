import numpy as np

### initialize PyJulia
# from .julia_jlcommand import jl

### Load CMBLensing.jl and spt3g
from spt3g.core import G3Units
from spt3g.maps import MapProjection, FlatSkyMap, MapPolConv
from spt3g.mapspectra.map_spectrum_classes import MapSpectrum2D, MapSpectrum1D
from spt3g.mapspectra.basicmaputils import get_fft_scale_fac, map_to_ft
from spt3g.lensing.map_spec_utils import MapSpectraTEB

jl("""
using CMBLensing
using CMBLensing: unfold
using Random

function maybe_int(x)
    i = round(Int, x)
    i ≈ x ? i : x
end
""")


jl("""


)


### Conversion functions
def FlatMap(f):
    jl("FlatMap($f, θpix=maybe_int($f.res / $G3Units.arcmin))")
    FlatMap(py"$p", py"$p.res")*60))
end

function FlatFourier(p::PyObject)
    @assert pytypeof(p) == py"MapSpectrum2D"
    scale_factor = py"get_fft_scale_fac(parent=$p.parent)"
    julia_fft = py"$p.get_complex()"[1:end÷2+1,:] * scale_factor
    FlatFourier(julia_fft, θpix=maybe_int(py"$p.parent.res / G3Units.arcmin"))
end

function FlatSkyMap(f)
    dx = fieldinfo(f).θpix
    py"FlatSkyMap($(Map(f).Ix).copy(order='C'), $dx*G3Units.arcmin, proj=MapProjection.ProjZEA)"o
end
MapFrame(f, keys="QU") = py"{k : $(FlatSkyMap(f[jl_keys[k]])) for k in keys}"o


class ObsCMBLensing:
    """
    Wraps CMBLensing's 'DataSet' interface in 3G Lensing's 'Obs' interface.
    This a just a general template that can be subclassed as needed.
    Work in progress.
    """

    def __init__(self, ds):

        self.ds = ds

        self.Nside = jl("fieldinfo($self.ds.d).Nside")
        self.shape = (self.Nside, self.Nside)
        self.res  = jl("fieldinfo($self.ds.d).θpix") * G3Units.arcmin


        self.pixel_mask = jl("diag($self.ds.M.b)[:Qx]")
        self.hash = {} # CinvFilt expects this to exist

    def get_dat_tqu(self):
        """
        Return the data.
        """
        return toFrame(jl("$self.ds.d"), "TQU", constructor, G3Units.uK)


    def get_sim(self, isim, keys="TQU", constructor=toFlatSkyMap):
        """
        Return a simulated data consistent with spt3g's data model: `d = B*f̃ + n`.
        """

        d = jl("""
            ds = $self.ds
            Random.seed!($isim)
            @unpack f,ϕ,n = resimulate(ds, return_truths=true)
            @unpack B,L = ds
            B * L(ϕ) * f + n
        """)

        return toFrame(d, keys, constructor, G3Units.uK)


    def get_sim_sky(self, isim, keys="TQU", constructor=toFlatSkyMap):
        """
        Return a simulated sky signal consistent with spt3g's data model: `d = B*f̃`.
        """

        d = jl("""
            ds = $self.ds
            Random.seed!($isim)
            @unpack f,ϕ,n = resimulate(ds, return_truths=true)
            @unpack B,L = ds
            B * L(ϕ) * f
        """)

        return toFrame(d, keys, constructor, G3Units.uK)

    get_sim_tqu     = lambda self, i: self.get_sim(i, "TQU")
    get_sim_tqu_sky = lambda self, i: self.get_sim_sky(i, "TQU")

    get_sim_teb     = lambda self, i: self.get_sim(i, "TEB", toMapSpectrum2D)
    get_sim_teb_sky = lambda self, i: self.get_sim_sky(i, "TEB", toMapSpectrum2D)
