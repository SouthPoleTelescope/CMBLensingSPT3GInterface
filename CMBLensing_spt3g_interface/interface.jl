using PyCall
import PyCall: PyObject
using CMBLensing
using CMBLensing: unfold, FlatIEB, FlatIQU

py"""
import numpy as np
from spt3g.core import G3Units
from spt3g.maps import MapProjection, FlatSkyMap, MapPolConv
from spt3g.mapspectra.map_spectrum_classes import MapSpectrum2D, MapSpectrum1D
from spt3g.mapspectra.basicmaputils import get_fft_scale_fac, map_to_ft
from spt3g.lensing.map_spec_utils import MapSpectraTEB
"""

Base.getindex(f::FlatField, s::String) = f[s == "T" ? :I : Symbol(s)]

"""
Return an empty FlatSkyMap "parent" that can be populated
with pixel values or passed to a MapSpectrum
"""
function Parent(f)
    @unpack Nside, θpix = fieldinfo(f)

    py"""
    parent = FlatSkyMap(
        x_len       = $Nside,
        y_len       = $Nside,
        res         = $θpix * G3Units.arcmin,
        weighted    = False,
        proj        = MapProjection.ProjNone,
        flat_pol    = True,
        pol_conv    = MapPolConv.IAU)
    """

    py"parent"o
end



"""
    FlatSkyMap(f::FlatField)

Convert a CMBLensing FlatField to a 3G FlatSkyMap.
"""
function FlatSkyMap(f::FlatField)
    flatskymap = Parent(f)

    py"""
    np.asarray($flatskymap)[:] = $(Map(f).Ix)[::-1, :]
    """

    flatskymap
end
PyObject(f::FlatMap) = FlatSkyMap(f)



"""
    MapSpectrum2D(f::FlatField)

Convert a CMBLensing FlatField to a 3G MapSpectrum2D,
applying the appropriate scale factor.
"""
function MapSpectrum2D(f::FlatField)
    # flip the fft in the y direction, excluding first row if
    fft = unfold(Fourier(f).Il)[:, 1:end÷2+1]
    fft_flip = @view fft[(1 + size(fft)[2] % 2) : end, :]
    fft_flip .= reverse(fft_flip, dims=1)

    parent = Parent(f)
    scale_factor = py"get_fft_scale_fac(parent=$parent)"

    py"MapSpectrum2D($parent, $fft / $scale_factor)"o
end
PyObject(f::FlatFourier) = MapSpectrum2D(f)



"""
    Frame(f::FieldTuple, keys, constructor::Function=identity; mult=1)

Convert a CMBLensing FieldTuple to a dictionary (a "Frame" in 3G parlance)
by calling `constructor` on the FieldTuple indexed by each of the
specified `keys`. An optional multiplicative factor `mult` may also be applied.
If the constructor is not specified, the output frame will be in the same basis
as the FieldTuple.

Parameters
----------
f: CMBLensing.jl FieldTuple
    input field to be converted to a Frame.

keys["TQU"]: iterable of characters
    keys of resulting Frame. Must be in "TQUEB".

constructor[toFlatSkyMap]: function
    function used to convert each CMBLensing map to a 3G software
    object. can be one of [toFlatSkyMap, toMapSpectrum2D].

mult[1]: number, ndarray, FlatSkyMap/MapSpectrum2D
    each value in the output Frame will be multiplied by `mult`.
    useful for applying units, masking etc.
"""

# full constructor
function Frame(f::FieldTuple, keys, constructor::Function=identity; mult=1)
    frame = py"{}"o
    for k in (string(k) for k in keys)
        set!(frame, k, constructor(f[k] * mult))
    end
    frame
end

# default keys for various FieldTuples
Frame(f::FlatIEB, con::Function=identity; mult=1) = Frame(f, "TEB", con, mult=mult)
Frame(f::FlatIQU, con::Function=identity; mult=1) = Frame(f, "TQU", con, mult=mult)
Frame(f::FlatEB,  con::Function=identity; mult=1) = Frame(f, "EB",  con, mult=mult)
Frame(f::FlatQU,  con::Function=identity; mult=1) = Frame(f, "QU",  con, mult=mult)

# jl -> py autoconvert
PyObject(f::FieldTuple) = Frame(f)



"""
Convert a CMBLensing FlatS02 to a 3G MapSpectraTEB.
"""
function MapSpectraTEB(f; mult=1)
    frame = Frame(f, "TEB", MapSpectrum2D, mult=mult)
    py"MapSpectraTEB($frame)"o
end
