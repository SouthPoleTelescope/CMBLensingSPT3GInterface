using PyCall
using CMBLensing
using CMBLensing: unfold
import CMBLensing: FlatMap, FlatFourier 

py"""
from spt3g.core import G3Units
from spt3g.maps import FlatSkyMap, MapProjection
from spt3g.mapspectra.map_spectrum_classes import MapSpectrum2D
from spt3g.mapspectra.basicmaputils import get_fft_scale_fac
"""

jl_keys = Dict(String(k) => Symbol(k) for k in ["Q", "U", "E", "B"])
jl_keys["T"] = :I

function FlatMap(p::PyObject)
    @assert pytypeof(p) == py"FlatSkyMap"
    FlatMap(py"$p", θpix=maybe_int(rad2deg(py"$p.res")*60))
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

function maybe_int(x)
    i = round(Int, x)
    i ≈ x ? i : x
end
