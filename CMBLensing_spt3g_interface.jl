using PyCall
using CMBLensing
using CMBLensing: unfold, fold
import CMBLensing: FlatMap, FlatFourier 

py"""
from spt3g.core import G3Units
from spt3g.coordinateutils import FlatSkyMap, MapProjection
from spt3g.mapspectra.basicmaputils import get_fft_scale_fac
"""

jl_keys = Dict(String(k) => Symbol(k) for k in ["Q", "U", "E", "B"])
jl_keys["T"] = :I

function FlatMap(p::PyObject)
    @assert pytypeof(p) == py"FlatSkyMap"
    FlatMap(py"$p", θpix=rad2deg(py"$p.res")*60)
end

function FlatFourier(p::PyObject)
    error("Not implemented")
    @assert pytypeof(p) == py"MapSpectrum2D"
    scale_factor = py"get_fft_scale_fac($p.res, $p.map_nx, $p.map_ny)"
    julia_fft = fold(py"$p")[....] * scale_factor
    FlatFourier(julia_fft, θpix=py"$p.res / G3Units.arcmin")
end

function FlatSkyMap(f)
    dx = fieldinfo(f).θpix
    py"FlatSkyMap($(Map(f).Ix).copy(order='C'), $dx*G3Units.arcmin, proj=MapProjection.ProjZEA)"o
end

MapFrame(f, keys="QU") = py"{k : $(FlatSkyMap(f[jl_keys[k]])) for k in keys}"o
