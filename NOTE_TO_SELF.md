In order to convert from musicxml to lilypond we rely on the script
"musicxml2ly", which is part of lilypond.

However, this script does not include the following features, which
are necessary here:

* microtones
* different notehead sizes

These features are implemented as modifications within two files:

* musicxml2ly.py
* musicexp.py

Since we want to distribute these modifications along this package, we
must include them in our tree. These scripts have not been updated
upstream and are still based on python 2. Including python 2 files in
a python 3 package is troublesome. Thus, the included files here are
included as data as follows:

1) The original files lie at sndscribe_xml2ly
2) They are copyied to sndscribe/xml2ly as `.py_` files. This is done
in order to avoid being interpreted as python files. If they are
included as python files they fail byte-compilation, since they are
not python 3 files
3) When the package is first imported, these files are copied to
$USERDATAFOLDER/sndscribe/xml2ly and the extension is changed back to
.py
4) During setup.py any old python 2 scripts are removed in order to
be copyied again after reinstallation


    
