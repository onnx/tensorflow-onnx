"""
tool to list tensorflow op to onnx op mapping in markdown
"""

import argparse
import inspect
import re

from collections import OrderedDict

from tf2onnx.handler import tf_op

parser = argparse.ArgumentParser()
parser.add_argument('filename',
                    type=str,
                    help="Path to the documentation file.")
args = parser.parse_args()


LATEST_OPSET = {
    "": 10, # default domain
    "com.microsoft": 1 # microsoft domain
}


def is_unsupported(func_body):
    source_code = inspect.getsource(func_body)
    if re.findall(r'raise NotImplementedError', source_code):
        return True
    return False


with open(args.filename, 'w+') as doc_file:
    doc_file.write("## `tf2onnx` Support Status\n")

    for domain, opsets in tf_op.get_opsets().items():
        comment = "(default domain)" if domain == "" else ""
        doc_file.write("### Domain: \"{}\" {}\n".format(domain, comment))
        doc_file.write("| Tensorflow Op | Convertible to ONNX Op Versions |\n")
        doc_file.write("| ------------- | ------------------------------- |\n")

        # Collect a mapping from tf ops to supported handler versions.
        tf_op_to_versions = OrderedDict()
        # Some op with NotImplementedError in it is unsupported.
        unsupported_cases = OrderedDict()
        for opset in opsets:
            for name, func in opset.items():
                handler_ver = int(func[0].__name__.replace("version_", ""))
                if name not in tf_op_to_versions or handler_ver < tf_op_to_versions[name]:
                    tf_op_to_versions[name] = handler_ver

                if is_unsupported(func[0]):
                    if name in unsupported_cases:
                        unsupported_cases[name].append(handler_ver)
                    else:
                        unsupported_cases[name] = [handler_ver]

        # Document support status according to the gathered mapping.
        for tf_op, supported_versions in tf_op_to_versions.items():
            if supported_versions < LATEST_OPSET[domain]:
                version_text = "{} ~ {}".format(supported_versions, LATEST_OPSET[domain])
                if tf_op in unsupported_cases:
                    version_text += " (Except {})".format(
                        ','.join(str(v) for v in unsupported_cases[tf_op])
                    )
            else:
                version_text = "{}".format(LATEST_OPSET[domain])
            doc_file.write("| {} | {} |\n".format(tf_op, version_text))
