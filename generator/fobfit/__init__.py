"""FobFit: a snap-on turning jacket for a car key, generated from a 3D scan of that key.

Modules
  geom      mesh conversions, iso-surfaces, point-to-triangle distance
  scan      loading, units, cleaning, key frame detection, caliper check
  cavity    signed distance field of the key and the jacket body (cavity, lips, keep-out zones), fit coupon
  wings     wing styles (Wing, Paddle, Loop); independent of the cavity code
  checks    verification of a finished jacket against a key
  report    report, deviation map and renders
  slicer    Bambu Studio command-line slicing (H2S, PETG)
  pipeline  the whole run, used by the window and by the command line
"""
__version__ = '1.0'
