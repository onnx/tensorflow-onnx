frozen=tests/models/fc-layers/frozen.pb
output=/tmp/model.onnx
input_names=X:0
output_names=output:0
output_names1=output

python -m tf2onnx.convert --input $frozen --inputs $input_names --outputs $output_names --output $output $@
