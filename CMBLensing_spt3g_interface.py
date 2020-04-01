### initialize PyJulia
import sys
import julia
from julia_jlcommand import jl

jl("""
using CMBLensing
using CMBLensing: unfold
using Random

jl_keys = Dict(String(k) => Symbol(k) for k in ["Q", "U", "E", "B"])
jl_keys["T"] = :I
""")

import numpy as np

### Load CMBLensing.jl and spt3g
from spt3g.core import G3Units
uK = G3Units.uK
arcmin = G3Units.arcmin

from spt3g.coordinateutils import FlatSkyMap, MapProjection
from spt3g.mapspectra.map_spectrum_classes import MapSpectrum2D, MapSpectrum1D
from spt3g.mapspectra.basicmaputils import get_fft_scale_fac
from spt3g.lensing.map_spec_utils import MapSpectraTEB

### Conversion functions

def toMapSpectrum2D(f):
    """
    Convert a CMBLensing FlatField to a 3G MapSpectrum2D,
    applying the appropriate scale factor.
    """
    map_nx = jl("fieldinfo($f)").Nside
    dx     = jl("fieldinfo($f)").θpix*arcmin
    fft    = jl("unfold(Fourier($f).Il)[:,1:end÷2+1]").copy(order='C')
    scale_factor = get_fft_scale_fac(dx, map_nx, map_nx)

    return MapSpectrum2D(map_nx, dx, fft / scale_factor)


def toFlatSkyMap(f):
    """
    Convert a CMBLensing FlatField to a 3G FlatSkyMap.
    """
    return FlatSkyMap(
        jl("Map($f).Ix").copy(order='C'),
        jl("fieldinfo($f)").θpix*arcmin,
        proj = MapProjection.ProjZEA
    )

def toMapSpectraTEB(f):
    """
    Convert a CMBLensing FlatS02 to a 3G MapSpectraTEB.
    """
    return MapSpectraTEB(toFrame(f, "TEB", toMapSpectrum2D, uK))

def toFrame(f, keys="TQU", constructor=toFlatSkyMap, units=1):
    """
    Convert a CMBLensing FieldTuple to a dictionary (a "Frame" in 3G parlance)
    by calling `constructor` on the FieldTuple indexed by each of the specified
    `keys`.
    By default, this constructs a Frame of TQU FlatSkyMaps.
    """
    return {k : constructor(jl("$f[jl_keys[$k]]")) * units for k in keys}

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

    def get_dat_tqu(self, keys="TQU", constructor=toFlatSkyMap):
        """
        Return the data.
        """
        return toFrame(jl("$self.ds.d"), keys, constructor, uK)


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

        return toFrame(d, keys, constructor, uK)


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

        return toFrame(d, keys, constructor, uK)

    get_sim_tqu     = lambda self, i: self.get_sim(i, "TQU")
    get_sim_tqu_sky = lambda self, i: self.get_sim_sky(i, "TQU")

    get_sim_teb     = lambda self, i: self.get_sim(i, "TEB", toMapSpectrum2D)
    get_sim_teb_sky = lambda self, i: self.get_sim_sky(i, "TEB", toMapSpectrum2D)
