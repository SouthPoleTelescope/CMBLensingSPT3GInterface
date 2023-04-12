module CMBLensingSPT3GInterface

export FlatSkyMap, MapSpectrum2D, Frame, MapSpectraTEB, @py_str, unitless

using AbstractFFTs
using CMBLensing
using CMBLensing: FlatIEB, FlatIQU, basis, SpatialBasis
using DataStructures
using Lazy
using PythonCall

@init @eval @pyexec """
import numpy as np
from spt3g.core import G3Units, G3TimestreamUnits, G3Frame
from spt3g.maps import MapProjection, FlatSkyMap, MapPolConv
from spt3g.mapspectra.map_spectrum_classes import MapSpectrum2D, MapSpectrum1D
from spt3g.mapspectra.basicmaputils import get_fft_scale_fac, map_to_ft
from spt3g.lensing.map_spec_utils import MapSpectraTEB
""" => (
    np, G3Units, G3TimestreamUnits, G3Frame, MapProjection, 
    FlatSkyMap, MapPolConv, MapSpectrum2D, MapSpectrum1D, 
    get_fft_scale_fac, map_to_ft, MapSpectraTEB
)

Base.getindex(f::FlatField, s::String) = f[s == "T" ? :I : Symbol(s)]


# """
#     similar_FlatSkyMap(f::FlatField)

# Return an empty FlatSkyMap with metadata similar to `f` that can be
# populated with pixel values or passed to a MapSpectrum as the "parent".
# """
# function similar_FlatSkyMap(f::FlatField; units)

#     @unpack Ny, Nx, θpix, rotator = fieldinfo(f)
#     rotator[1] == rotator[3] == 0 || @error("rotator != (0, x, 0) not yet handled.")
#     delta_center = deg2rad(rotator[2]-90)

#     @pyeval (Nx,Ny,θpix,delta_center,units) => """
#     FlatSkyMap(
#         x_len        = Nx,
#         y_len        = Ny,
#         res          = θpix * G3Units.arcmin,
#         delta_center = delta_center,
#         weighted     = False,
#         proj         = MapProjection.ProjZEA,
#         flat_pol     = True,
#         pol_conv     = MapPolConv.IAU,
#         units        = None if units == 1 else G3TimestreamUnits.Tcmb
#     )
#     """

# end


function get_proj_kwargs(f::Py)
    θpix = pyconvert(Number, f.res / G3Units.arcmin)
    θpix = (θpix ≈ round(Int, θpix)) ? round(Int, θpix) : θpix
    pyconvert(Number, f.alpha_center) == 0 || @error("alpha_center != 0 not yet handled.") 
    rotator = (0., 90 + rad2deg(pyconvert(Number, f.delta_center)), 0.)
    (; θpix, rotator)
end


# wraps a CMBLensing field and indicates that auto-conversion of this object is
# to assume that the field is unitless, i.e. `units=1`
struct Unitless{F}
    f::F
end
unitless(f) = Unitless(f)


# warn about non-ZEA projections
function check_proj(f::Py)
    if pyconvert(Bool, f.proj != MapProjection.ProjZEA)
        @warn """
        CMBLensing is only designed to handle ProjZEA maps. 
        Using maps in other projections may violate assumptions made by CMBLensing, and if you convert back spt3g_software the projection will be changed to ProjZEA.
        To remove this warning, you can set `flatskymap.proj = spt3g.maps.MapProjection.ProjZEA` before converting.
        """
    end
end



# ### FlatMap <--> FlatSkyMap
# ###########################

# ### jl -> py
# ############
# """
#     FlatSkyMap(f::FlatMap; units=μK)

# Convert a CMBLensing FlatMap to a 3G FlatSkyMap. The FlatMap is assumed to have
# units given by `units`. 
# """
# function FlatSkyMap(f::FlatMap; units=μK)
#     flatskymap = similar_FlatSkyMap(f; units)
#     @pyexec (flatskymap, arr = f.Ix .* units) => "np.copyto(np.asarray(flatskymap), arr)"
#     flatskymap
# end

# PyCall.PyObject(f::FlatMap; kwargs...) = FlatSkyMap(f; kwargs...)
# PyCall.PyObject(u::Unitless{<:FlatMap}) = FlatSkyMap(u.f, units=1)


### py -> jl
############

function pyconvert_rule_FlatSkyMap(::Type{<:Field}, flatskymap::Py; units=nothing)
    check_proj(flatskymap)
    if units === nothing
        units = pyconvert(Bool, flatskymap.units == G3TimestreamUnits.Tcmb) ? G3Units.uK : 1
    end
    field = FlatMap(pyconvert(Array, np.asarray(flatskymap) / units); get_proj_kwargs(flatskymap)...)
    return PythonCall.pyconvert_return(field)
end

@init PythonCall.pyconvert_add_rule("spt3g.maps:FlatSkyMap", Field, pyconvert_rule_FlatSkyMap)


# ### FlatFourier <--> MapSpectrum2D
# #################################

# ### jl -> py
# ############
# """
#     MapSpectrum2D(f::FlatFourier)

# Convert a CMBLensing FlatFourier to a 3G MapSpectrum2D. 
# """
# function MapSpectrum2D(f::FlatFourier; units=nothing)
#     parent = similar_FlatSkyMap(f, units=units)
#     if units == nothing
#         units = 1/py"get_fft_scale_fac(parent=$parent)"
#         py"""
#         parent.units = G3TimestreamUnits.Tcmb
#         0"""
#     end
#     Il = f[:Il, full_plane=true][:,1:end÷2+1] * units
#     py"MapSpectrum2D($parent, $Il.copy(order='C'))"o
# end

# PyCall.PyObject(f::FlatFourier; kwargs...) = MapSpectrum2D(f; kwargs...)
# PyCall.PyObject(u::Unitless{<:FlatFourier}) = MapSpectrum2D(u.f, units=1)


# ### py -> jl
# ############

function pyconvert_rule_MapSpectrum2D(::Type{<:Field}, mapspectrum2d::Py; units=nothing)
    flatskymap = mapspectrum2d.parent
    check_proj(flatskymap)
    if units == nothing
        units = pyconvert(Bool, flatskymap.units == G3TimestreamUnits.Tcmb) ? 1/get_fft_scale_fac(parent=flatskymap) : 1
    end
    Il = pyconvert(Array, np.asarray(mapspectrum2d.get_complex())/units)[1:end÷2+1,:]
    field = FlatFourier(Il; get_proj_kwargs(flatskymap)..., Ny=pyconvert(Int,flatskymap.shape[0]))
    return PythonCall.pyconvert_return(field)
end
@init PythonCall.pyconvert_add_rule("spt3g.mapspectra.map_spectrum_classes:MapSpectrum2D", Field, pyconvert_rule_MapSpectrum2D)

# @init pytype_mapping(py"MapSpectrum2D", FlatFourier)
# Base.convert(::Type{FlatFourier}, f::PyObject) = FlatFourier(f)


### Union{FlatS2,FlatS02} <--> Frame
####################################

F_pykey_mapping = OrderedDict(
    FlatIQUFourier  => ("T", "Q", "U"),
    FlatIQUMap      => ("T", "Q", "U"),
    FlatIEBFourier  => ("T", "E", "B"),
    FlatIEBMap      => ("T", "E", "B"),
    FlatQUFourier   =>      ("Q", "U"),
    FlatQUMap       =>      ("Q", "U"),
    FlatEBFourier   =>      ("E", "B"),
    FlatEBMap       =>      ("E", "B")
)
function pykeys(f::Union{FlatS2,FlatS02})
    for (F,pykeys) in F_pykey_mapping
        f isa F && return pykeys
    end
end

# ### jl -> py
# ############

# """
#     Frame(f::Union{FlatS2,FlatS02}, keys=pykeys(f))

# Convert a CMBLensing FlatS2 or FlatS02 to a 3G Frame (i.e. a dict). If `keys` is
# supplied, should be some subset of "TEBQU" to include in the dict. 
# """
# function Frame(f::Union{FlatS2,FlatS02}, keys=pykeys(f); kwargs...)
#     frame = py"{}"o
#     for k in (string(k) for k in keys)
#         set!(frame, k, PyObject(f[k]; kwargs...))
#     end
#     frame
# end

# # FlatIEBFourier is the only FlatField-like object which has a
# # corresponding type on the 3G side (ie a MapSpectraTEB)
# PyCall.PyObject(f::FlatIEBFourier; kwargs...)  = MapSpectraTEB(f; kwargs...)
# PyCall.PyObject(u::Unitless{<:FlatIEBFourier}) = MapSpectraTEB(u.f, units=1)
# # for everything else, convert to generic "Frame" which is just a dict
# PyCall.PyObject(f::Union{FlatS2,FlatS02}; kwargs...)  = Frame(f; kwargs...)
# PyCall.PyObject(u::Unitless{<:Union{FlatS2,FlatS02}}) = Frame(u.f, units=1)

### py -> jl
############

function pyconvert_rule_Frame(::Type{<:Field}, frame::Py; units=nothing)
    for (F, pykeys) in F_pykey_mapping
        F3G = (basis(F) <: SpatialBasis{Map}) ? FlatSkyMap : MapSpectrum2D
        if all(pyisinstance(frame.get(k,nothing), F3G) for k in pykeys)
            field = F((pyconvert(Field, frame[k]) for k in pykeys)...)
            return PythonCall.pyconvert_return(field)
        end
    end
    return PythonCall.pyconvert_unconverted()
end

@init PythonCall.pyconvert_add_rule("spt3g.core:G3Frame", Field, pyconvert_rule_Frame)
@init PythonCall.pyconvert_add_rule("builtins:dict",      Field, pyconvert_rule_Frame)


# ### FlatIEBFourier <--> MapSpectraTEB
# #####################################

# ### jl -> py
# ############
# """
# Convert a CMBLensing FlatS02 to a 3G MapSpectraTEB.
# """
# function MapSpectraTEB(f::FlatIEBFourier; kwargs...)
#     frame = Frame(f; kwargs...)
#     py"MapSpectraTEB($frame)"o
# end

# ### py -> jl
# ############

# function CMBLensing.FlatIEBFourier(f::PyObject)
#     if pyisinstance(f, py"MapSpectraTEB")
#         FlatIEBFourier(py"[$f[k] for k in 'TEB']"...)
#     else
#         error("Can't convert a Python object of type $(pytypeof(f)) to a FlatIEBFourier.")
#     end
# end

# @init pytype_mapping(py"MapSpectraTEB ", FlatIEBFourier)
# Base.convert(::Type{FlatIEBFourier}, f::PyObject) = FlatIEBFourier(f)


end