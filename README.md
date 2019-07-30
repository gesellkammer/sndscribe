# sndscribe

**sndscribe** translates a spectrum of a soundfile into musical notation.

A spectrum is the result of partial tracking analysis, done via
[sndtrck]
This is done in a series of steps:

* spectral analysis, using [sndtrck]
* translation of partial tracking information to musicxml, with the
  possibility of using microtones and dynamics
* automatic conversion of the generated musicxml to pdf via lilypond
  
# Dependencies

* csound >= 6.12 (for playback)


# installation

`pip install sndscribes`

That should install all the python dependencies needed


**NB**: only python >= 3.7 supported
  

[sndtrck]: https://github.com/gesellkammer/sndtrck
[csound-plugins]: https://github.com/csound-plugins/csound-plugins