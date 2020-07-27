using CMBLensing
using CMBLensingSPT3GInterface
using PyCall
using Test

py"""
from spt3g.maps import FlatSkyMap
from spt3g.mapspectra.map_spectrum_classes import MapSpectrum2D
from spt3g.lensing.map_spec_utils import MapSpectraTEB
"""

@testset "Conversions (Nside=$Nside)" for Nside=[32,33]

    f = FlatIQUMap(rand(Nside,Nside),rand(Nside,Nside),rand(Nside,Nside),θpix=2)

    # CMBLensing fields end up the right type on the Python side
    @test py"isinstance($(       Map(f[:I])), FlatSkyMap)"
    @test py"isinstance($(   Fourier(f[:I])), MapSpectrum2D)"
    @test py"isinstance($(IEBFourier(f)),     MapSpectraTEB)"
    @test py"isinstance($(    IQUMap(f)),     dict)"
    
    # the field is unchanged after a roundtrip jl -> py -> jl
    @test py"$(       Map(f[:I]))" ≈        Map(f[:I])
    @test py"$(   Fourier(f[:I]))" ≈    Fourier(f[:I])
    @test py"$(IEBFourier(f))"     ≈ IEBFourier(f)
    @test py"$(    IQUMap(f))"     ≈     IQUMap(f)

    # bad conversions throw custom errors
    @test_throws ErrorException    FlatFourier(PyObject(f))
    @test_throws ErrorException        FlatMap(PyObject(f))
    @test_throws ErrorException FlatIEBFourier(PyObject(f[:I]))

end