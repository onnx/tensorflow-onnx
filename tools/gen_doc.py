"""
tool to list tensorflow op to onnx op mapping in markdown
"""

import argparse

from collections import OrderedDict

from tf2onnx.handler import tf_op

parser = argparse.ArgumentParser()
parser.add_argument('filename',
                    type=str,
                    help="Path to the documentation file.")
args = parser.parse_args()

with open(args.filename, 'w+') as doc_file:
    doc_file.write("## `tf2onnx` Support Status\n")

    for domain, opsets in tf_op.get_opsets().items():
        comment = "(default domain)" if domain == "" else ""
        doc_file.write("### Domain: \"{}\" {}\n".format(domain, comment))
        doc_file.write("| Tensorflow Op | Convertible to ONNX Op Versions |\n")
        doc_file.write("| ------------- | ------------------------------- |\n")

        # Collect a mapping from tf ops to supported handler versions.
        tf_op_to_versions = OrderedDict()
        for opset in opsets:
            for name, func in opset.items():
                handler_ver = func[0].__name__.replace("version_", "")
                if name in tf_op_to_versions:
                    tf_op_to_versions[name].append(handler_ver)
                else:
                    tf_op_to_versions[name] = [handler_ver]

        # Document support status according to the gathered mapping.
        for tf_op, supported_versions in tf_op_to_versions.items():
            doc_file.write("| {} | {} |\n".format(tf_op, supported_versions))
