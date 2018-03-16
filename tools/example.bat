set frozen=tests/models/fc-layers/frozen.pb
set output=/tmp/model.onnx
set input_names=X:0
set input_names1=X
set output_names=output:0
set output_names1=output

python -c "import tensorflow as tf; print(tf.sysconfig.get_lib()+'/python/tools')" >toolsdir
set /p toolsdir=<toolsdir
del toolsdir

python tf2onnx\convert.py --input %frozen% --inputs %input_names% --outputs %output_names% --output %output% %1 %2