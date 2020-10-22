using CMBLensing
using CMBLensingSPT3GInterface
import CMBLensingSPT3GInterface: μK
using PyCall
using Test

py"""
from spt3g.maps import FlatSkyMap
from spt3g.mapspectra.map_spectrum_classes import MapSpectrum2D
from spt3g.mapspectra.basicmaputils import map_to_ft
from spt3g.lensing.map_spec_utils import MapSpectraTEB
from numpy.testing import assert_allclose
"""

@testset "Conversions (Nside=$Nside)" for Nside=[32,33]

    f = FlatIQUMap(rand(Nside,Nside),rand(Nside,Nside),rand(Nside,Nside),θpix=2)

    # CMBLensing fields end up the right type on the Python side
    @test py"isinstance($(       Map(f[:I])), FlatSkyMap)"
    @test py"isinstance($(   Fourier(f[:I])), MapSpectrum2D)"
    @test py"isinstance($(IEBFourier(f)),     MapSpectraTEB)"
    @test py"isinstance($(    IQUMap(f)),     dict)"

    
    # make sure the type is the same after a roundtrip
    # (this checks if autoconvert is failing silently)
    @test py"$(Map(f[:I]))" isa FlatMap
    @test py"$(Fourier(f[:I]))" isa FlatFourier

    # the field is unchanged after a roundtrip jl -> py -> jl, 
    # both when converted in μK units:
    @test py"$(       Map(f[:I]))" ≈        Map(f[:I])
    @test py"$(   Fourier(f[:I]))" ≈    Fourier(f[:I])
    @test py"$(   Fourier(f[:I]))" ≈    Fourier(f[:I])
    @test py"$(IEBFourier(f))"     ≈ IEBFourier(f)
    @test py"$(    IQUMap(f))"     ≈     IQUMap(f)

    # or unitless:
    @test py"$(unitless(       Map(f[:I])))" ≈        Map(f[:I])
    @test py"$(unitless(   Fourier(f[:I])))" ≈    Fourier(f[:I])
    @test py"$(unitless(IEBFourier(f)))"     ≈ IEBFourier(f)
    @test py"$(unitless(    IQUMap(f)))"     ≈     IQUMap(f)

    # cross-basis on spt3g_software side
    @test py"$(Fourier(f[:I])).get_rmap()" ≈ Map(f[:I])
    @test py"map_to_ft($(Map(f[:I])))"     ≈ Fourier(f[:I])

    # applying units
    @test py"$(FlatSkyMap(Map(f[:I]), units=μK^2))" ≈ Map(py"$(MapSpectrum2D(Fourier(f[:I]), units=μK^2))")
    @test py"$(Map(f[:I])) * $μK**2"                ≈ Map(py"$(Fourier(f[:I]))            * $μK**2")
    @test py"$(Map(f[:I])) * $μK**2"                ≈     py"$(Fourier(f[:I])).get_rmap() * $μK**2"
    @test py"$(unitless(Map(f[:I]))) * $μK**2"      ≈ Map(py"$(unitless(Fourier(f[:I])))            * $μK**2")
    @test py"$(unitless(Map(f[:I]))) * $μK**2"      ≈     py"$(unitless(Fourier(f[:I]))).get_rmap() * $μK**2"

    # bad conversions throw custom errors
    @test_throws ErrorException    FlatFourier(PyObject(f))
    @test_throws ErrorException        FlatMap(PyObject(f))
    @test_throws ErrorException FlatIEBFourier(PyObject(f[:I]))

end