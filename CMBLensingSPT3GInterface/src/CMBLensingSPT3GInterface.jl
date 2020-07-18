module CMBLensingSPT3GInterface

export similar_FlatSkyMap, Frame, MapSpectraTEB
export @py_str


using PyCall
import PyCall: PyObject
using CMBLensing
using CMBLensing: unfold, FlatIEB, FlatIQU
using AbstractFFTs

function __init__()
    py"""
    import numpy as np
    from spt3g.core import G3Units
    from spt3g.maps import MapProjection, FlatSkyMap, MapPolConv
    from spt3g.mapspectra.map_spectrum_classes import MapSpectrum2D, MapSpectrum1D
    from spt3g.mapspectra.basicmaputils import get_fft_scale_fac, map_to_ft
    from spt3g.lensing.map_spec_utils import MapSpectraTEB
    0"""

    pytype_mapping(py"FlatSkyMap", FlatMap)
    pytype_mapping(py"MapSpectrum2D", FlatFourier)
    pytype_mapping(py"MapSpectraTEB ", FlatIEBFourier)
    pytype_mapping(py"dict", FieldTuple)
end

Base.getindex(f::FlatField, s::String) = f[s == "T" ? :I : Symbol(s)]


"""
    similar_FlatSkyMap(f::FlatField)

Return an empty FlatSkyMap with metadata similar to `f` that can be
populated with pixel values or passed to a MapSpectrum as the "parent".
"""
function similar_FlatSkyMap(f::FlatField)

    @unpack Nside, θpix = fieldinfo(f)

    py"""
    parent = FlatSkyMap(
        x_len       = $Nside,
        y_len       = $Nside,
        res         = $θpix * G3Units.arcmin,
        weighted    = False,
        proj        = MapProjection.ProjNone,
        flat_pol    = True,
        pol_conv    = MapPolConv.IAU
    )"""

    py"parent"o
    
end


"""
    flipy(f)

Flip a field in the y-direction.
"""
function flipy(f::F) where {F<:FlatMap}
    F(f.Ix[end:-1:1,:])
end
function flipy(f::F) where {N,F<:FlatFourier{<:Flat{N}}}
    # equivalent of shift by 1 map pixel in y direction
    shift = @. exp(2π * im * $(0:N÷2) / N)
    # for each Fourier frequency, the index of the negative frequency
    negky = circshift(ifftshift(reverse(fftshift(1:N))), iseven(N) ? 1 : 0)[1:end÷2+1]
    F(unfold(f.Il)[negky,:] .* shift)
end

function get_θpix(f::PyObject)
    θpix = py"$f.res/G3Units.arcmin"
    θpix ≈ round(Int,θpix) ? round(Int,θpix) : θ
end


### FlatMap <--> FlatSkyMap
###########################

### jl -> py
############
"""
    FlatSkyMap(f::FlatField)

Convert a CMBLensing FlatField to a 3G FlatSkyMap.
"""
function FlatSkyMap(f::FlatMap)
    flatskymap = similar_FlatSkyMap(f)

    py"""
    np.asarray($flatskymap)[:] = $(flipy(f).Ix)
    0"""

    flatskymap
end
PyObject(f::FlatMap) = FlatSkyMap(f)



### py -> jl
############

function CMBLensing.FlatMap(f::PyObject)
    @assert pytypeof(f) == py"FlatSkyMap"
    flipy(FlatMap(py"np.asarray($f)", θpix=get_θpix(f)))
end

Base.convert(::Type{FlatMap}, f::PyObject) = FlatMap(f)


### FlatFourier <--> MapSpectrum2D
#################################

### jl -> py
############
"""
    MapSpectrum2D(f::FlatField)

Convert a CMBLensing FlatField to a 3G MapSpectrum2D,
applying the appropriate scale factor.
"""
function MapSpectrum2D(f::FlatFourier)
    parent = similar_FlatSkyMap(f)
    scale_fac = py"get_fft_scale_fac(parent=$parent)"
    Il = unfold(flipy(f).Il)[:,1:end÷2+1] / scale_fac
    py"MapSpectrum2D($parent, $Il)"o
end
PyObject(f::FlatFourier) = MapSpectrum2D(f)

### py -> jl
############

function CMBLensing.FlatFourier(f::PyObject)
    @assert pytypeof(f) == py"MapSpectrum2D"
    scale_fac = py"get_fft_scale_fac(parent=$f.parent)"
    Il = py"np.asarray($f.get_complex())"[1:end÷2+1,:] * scale_fac
    flipy(FlatFourier(Il, θpix=get_θpix(py"$f.parent"o)))
end
Base.convert(::Type{FlatFourier}, f::PyObject) = FlatFourier(f)


### FieldTuple <--> FieldTuple
###############################


### jl -> py
############
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
function Frame(f::FieldTuple, keys)
    frame = py"{}"o
    for k in (string(k) for k in keys)
        set!(frame, k, f[k])
    end
    frame
end

# default keys for various FieldTuples
Frame(f::FlatIEB) = Frame(f, "TEB")
Frame(f::FlatIQU) = Frame(f, "TQU")
Frame(f::FlatEB) = Frame(f, "EB")
Frame(f::FlatQU) = Frame(f, "QU")

PyObject(f::FieldTuple) = Frame(f)

### py -> jl
############

function Base.convert(::Type{FieldTuple}, frame::PyObject)
    for (F, pykeys) in [
        (FlatQUFourier,       ("Q", "U")),
        (FlatQUMap,           ("Q", "U")),
        (FlatEBFourier,       ("E", "B")),
        (FlatEBMap,           ("E", "B")),
        (FlatIQUFourier, ("T", "Q", "U")),
        (FlatIQUMap,     ("T", "Q", "U")),
        (FlatIEBFourier, ("T", "E", "B")),
        (FlatIEBMap,     ("T", "E", "B"))]

        F3G = (F <: FlatFieldMap) ? py"FlatSkyMap" : py"MapSpectrum2D"
        if py"all(type(f) == $F3G for f in $frame.values())" && py"sorted($frame) == sorted($pykeys)"
            return F(py"[$frame[k] for k in $pykeys]"...)
        end
    end

    copy(PyDict(frame))
end


### FlatIEBFourier <--> MapSpectraTEB
#####################################

### jl -> py
############
"""
Convert a CMBLensing FlatS02 to a 3G MapSpectraTEB.
"""
function MapSpectraTEB(f::FlatIEBFourier)
    frame = Frame(f, "TEB")
    py"MapSpectraTEB($frame)"o
end

### py -> jl
############
function Base.convert(::Type{FlatIEBFourier}, f::PyObject)
    @assert pytypeof(f) == py"MapSpectraTEB"
    FlatIEBFourier(py"[$f[k] for k in 'TEB']"...)
end

end # module
