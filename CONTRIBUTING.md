<!--- SPDX-License-Identifier: Apache-2.0 -->

# Contributing

We're always looking for your help to fix bugs and improve the product. Create a pull request and we'll be happy to take a look.

# Checkin procedure
1. Fork the repo
2. git clone your fork
3. Create feature branch
4. Make and checkin your changes along with unit tests
5. git commit your changes
6. git push origin HEAD
7. To request merge into main send a pull request from the web ui
https://github.com/onnx/tensorflow-onnx.


New code *must* be accompanied by unit tests.

*Note*: After creating a pull request, you will see a build getting triggered right away. You may check if style check and unit tests are passing.


# Coding guidelines
Please see [Coding Conventions and Standards](http://google.github.io/styleguide/pyguide.html)

# Licensing guidelines
This project welcomes contributions and suggestions. The contributions require you to
agree the Developer Certificate of Origin (DCO) declaring that you have the right to,
and actually do, grant us the rights to use your contribution.

When you submit a pull request, a DCO-bot will automatically determine whether you need
to provide a DCO and decorate the PR appropriately.

You are ready to sign your code by using the `-s` flag during your commits.

```sh
git commit -s
```

The repository follows the [REUSE specification](https://reuse.software/spec/).
Every file must have copyright and license information. Most files get it from
`REUSE.toml`; files from other projects need their own notice. Run the same
check as CI when you add, rename, or copy in files, or change licensing or
attribution:

```sh
python -m pip install reuse==6.3.0
reuse lint
```


# Code of conduct
This project has adopted the [ONNX Code of Conduct](https://onnx.ai/codeofconduct.html).
For questions or to report violations, contact [conduct@lfai.foundation](mailto:conduct@lfai.foundation).
