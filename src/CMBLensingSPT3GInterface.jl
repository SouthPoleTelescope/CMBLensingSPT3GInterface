module CMBLensingSPT3GInterface

export FlatSkyMap, Frame, MapSpectraTEB, MapSpectrum2D, @py_str, unitless

using PyCall
using CMBLensing
using CMBLensing: unfold, FlatIEB, FlatIQU
using AbstractFFTs
using Lazy

@init py"""
import numpy as np
from spt3g.core import G3Units, G3TimestreamUnits
from spt3g.maps import MapProjection, FlatSkyMap, MapPolConv
from spt3g.mapspectra.map_spectrum_classes import MapSpectrum2D, MapSpectrum1D
from spt3g.mapspectra.basicmaputils import get_fft_scale_fac, map_to_ft
from spt3g.lensing.map_spec_utils import MapSpectraTEB
"""
@init global μK = py"G3Units.uK"
@init global Tcmb = py"G3TimestreamUnits.Tcmb"


Base.getindex(f::FlatField, s::String) = f[s == "T" ? :I : Symbol(s)]


"""
    similar_FlatSkyMap(f::FlatField)

Return an empty FlatSkyMap with metadata similar to `f` that can be
populated with pixel values or passed to a MapSpectrum as the "parent".
"""
function similar_FlatSkyMap(f::FlatField; units)

    @unpack Nside, θpix = fieldinfo(f)

    py"""
    parent = FlatSkyMap(
        x_len       = $Nside,
        y_len       = $Nside,
        res         = $θpix * G3Units.arcmin,
        weighted    = False,
        proj        = MapProjection.ProjNone,
        flat_pol    = True,
        pol_conv    = MapPolConv.IAU,
        units       = None if $units == 1 else G3TimestreamUnits.Tcmb
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


# wraps a CMBLensing field and indicates that auto-conversion of this object is
# to assume that the field is unitless, i.e. `units=1`
struct Unitless{F}
    f::F
end
unitless(f) = Unitless(f)



### FlatMap <--> FlatSkyMap
###########################

### jl -> py
############
"""
    FlatSkyMap(f::FlatMap; units=μK)

Convert a CMBLensing FlatMap to a 3G FlatSkyMap. The FlatMap is assumed to have
units given by `units`. 
"""
function FlatSkyMap(f::FlatMap; units=μK)
    flatskymap = similar_FlatSkyMap(f, units=units)
    py"np.copyto(np.asarray($flatskymap), $(flipy(f).Ix * units))"
    flatskymap
end

PyCall.PyObject(f::FlatMap; kwargs...) = FlatSkyMap(f; kwargs...)
PyCall.PyObject(u::Unitless{<:FlatMap}) = FlatSkyMap(u.f, units=1)


### py -> jl
############

function CMBLensing.FlatMap(f::PyObject; units=nothing)
    if pyisinstance(f, py"FlatSkyMap")
        if units == nothing
            units = (py"$f.units" == Tcmb) ? μK : 1
        end
        flipy(FlatMap(py"np.asarray($f)" / units, θpix=get_θpix(f)))
    else
        error("Can't convert a Python object of type $(pytypeof(f)) to a FlatMap.")
    end
end

@init pytype_mapping(py"FlatSkyMap", FlatMap)
Base.convert(::Type{FlatMap}, f::PyObject) = FlatMap(f)


### FlatFourier <--> MapSpectrum2D
#################################

### jl -> py
############
"""
    MapSpectrum2D(f::FlatFourier)

Convert a CMBLensing FlatFourier to a 3G MapSpectrum2D. 
"""
function MapSpectrum2D(f::FlatFourier; units=nothing)
    parent = similar_FlatSkyMap(f, units=units)
    if units == nothing
        units = 1/py"get_fft_scale_fac(parent=$parent)"
        py"""
        parent.units = G3TimestreamUnits.Tcmb
        0"""
    end
    Il = unfold(flipy(f).Il)[:,1:end÷2+1] * units
    py"MapSpectrum2D($parent, $Il)"o
end

PyCall.PyObject(f::FlatFourier; kwargs...) = MapSpectrum2D(f; kwargs...)
PyCall.PyObject(u::Unitless{<:FlatFourier}) = MapSpectrum2D(u.f, units=1)


### py -> jl
############

function CMBLensing.FlatFourier(f::PyObject; units=nothing)
    if pyisinstance(f, py"MapSpectrum2D")
        if units == nothing
            units = (py"$f.parent.units" == Tcmb) ? 1/py"get_fft_scale_fac(parent=$f.parent)" : 1
        end
        Il = py"np.asarray($f.get_complex())"[1:end÷2+1,:] / units
        flipy(FlatFourier(Il, θpix=get_θpix(py"$f.parent"o)))
    else
        error("Can't convert a Python object of type $(pytypeof(f)) to a FlatFourier.")
    end
end

@init pytype_mapping(py"MapSpectrum2D", FlatFourier)
Base.convert(::Type{FlatFourier}, f::PyObject) = FlatFourier(f)


### Union{FlatS2,FlatS02} <--> Frame
####################################

F_pykey_mapping = Dict(
    FlatQUFourier   =>      ("Q", "U"),
    FlatQUMap       =>      ("Q", "U"),
    FlatEBFourier   =>      ("E", "B"),
    FlatEBMap       =>      ("E", "B"),
    FlatIQUFourier  => ("T", "Q", "U"),
    FlatIQUMap      => ("T", "Q", "U"),
    FlatIEBFourier  => ("T", "E", "B"),
    FlatIEBMap      => ("T", "E", "B")
)
function pykeys(f::Union{FlatS2,FlatS02})
    for (F,pykeys) in F_pykey_mapping
        f isa F && return pykeys
    end
end

### jl -> py
############

"""
    Frame(f::Union{FlatS2,FlatS02}, keys=pykeys(f))

Convert a CMBLensing FlatS2 or FlatS02 to a 3G Frame (i.e. a dict). If `keys` is
supplied, should be some subset of "TEBQU" to include in the dict. 
"""
function Frame(f::Union{FlatS2,FlatS02}, keys=pykeys(f); kwargs...)
    frame = py"{}"o
    for k in (string(k) for k in keys)
        set!(frame, k, PyObject(f[k]; kwargs...))
    end
    frame
end

# FlatIEBFourier is the only FieldTuple-like object which has a
# corresponding type on the 3G side (ie a MapSpectraTEB)
PyCall.PyObject(f::FlatIEBFourier; kwargs...)  = MapSpectraTEB(f; kwargs...)
PyCall.PyObject(u::Unitless{<:FlatIEBFourier}) = MapSpectraTEB(u.f, units=1)
# for everything else, convert to generic "Frame" which is just a dict
PyCall.PyObject(f::Union{FlatS2,FlatS02}; kwargs...)  = Frame(f; kwargs...)
PyCall.PyObject(u::Unitless{<:Union{FlatS2,FlatS02}}) = Frame(u.f, units=1)

### py -> jl
############

# note: we are stealing the pytype_mapping for _all_ dicts here, so make sure to
# fall back to default behavior if the dict wasn't actually a frame, i.e. didn't
# contain any 3G field objects.
@init pytype_mapping(py"dict", FieldTuple)
function Base.convert(::Type{FieldTuple}, frame::PyObject)
    for (F, pykeys) in F_pykey_mapping
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
function MapSpectraTEB(f::FlatIEBFourier; kwargs...)
    frame = Frame(f; kwargs...)
    py"MapSpectraTEB($frame)"o
end

### py -> jl
############

function CMBLensing.FlatIEBFourier(f::PyObject)
    if pyisinstance(f, py"MapSpectraTEB")
        FlatIEBFourier(py"[$f[k] for k in 'TEB']"...)
    else
        error("Can't convert a Python object of type $(pytypeof(f)) to a FlatIEBFourier.")
    end
end

@init pytype_mapping(py"MapSpectraTEB ", FlatIEBFourier)
Base.convert(::Type{FlatIEBFourier}, f::PyObject) = FlatIEBFourier(f)




# This following is needed for PyCall versions before
# https://github.com/JuliaPy/PyCall.jl/pull/792 is merged

using Pkg
using PyCall: TypeTuple, pytype_queries, pyint_query, npy_bool, 
    pyfloat_query, pycomplex_query, pystring_query, pyfunction_query, 
    pydate_query, pydict_query, pyptr_query, pysequence_query, 
    pynothing_query, pymp_query, @return_not_None

if Pkg.dependencies()[Base.PkgId(PyCall).uuid].git_revision != "pytype_mapping_prec"

    function PyCall.pytype_query(o::PyObject, default::TypeTuple=PyObject)
        # TODO: Use some kind of hashtable (e.g. based on PyObject_Type(o)).
        #       (A bit tricky to correctly handle Tuple and other containers.)
        for (py,jl) in pytype_queries
            if pyisinstance(o, py)
                return jl
            end
        end
        @return_not_None pyint_query(o)
        pyisinstance(o, npy_bool) && return Bool
        @return_not_None pyfloat_query(o)
        @return_not_None pycomplex_query(o)
        @return_not_None pystring_query(o)
        @return_not_None pyfunction_query(o)
        @return_not_None pydate_query(o)
        @return_not_None pydict_query(o)
        @return_not_None pyptr_query(o)
        @return_not_None pysequence_query(o)
        @return_not_None pynothing_query(o)
        @return_not_None pymp_query(o)
        return default
    end

end


end # module
