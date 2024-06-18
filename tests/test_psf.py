def test_roman_psf():
    from RomanDESCForwardModelLightcurves import get_roman_psf

    band = "H158"
    sca = 17
    x = 200
    y = 400

    psf = get_roman_psf(band, sca, x, y)
    assert psf.shape == (45, 45)
