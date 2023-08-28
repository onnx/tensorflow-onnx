# SPDX-License-Identifier: Apache-2.0


"""Unit Tests for einsum decomposition."""

import unittest
import itertools
import numpy as np
from numpy.testing import assert_almost_equal
from onnxruntime import InferenceSession
from tf2onnx.optimizer.einsum_optimizer import (
    analyse_einsum_equation, decompose_einsum_equation, EinsumSubOp)
from backend_test_base import Tf2OnnxBackendTestBase
from common import check_opset_min_version


class TestEinsum(Tf2OnnxBackendTestBase):
    "unit tests for einsum optimizer"

    def assert_raise(self, fct, exc_type):
        try:
            fct()
        except exc_type:
            return
        raise AssertionError("%r was not raised." % exc_type)

    def apply_einsum_sequence(self, seq, *inputs):
        providers = ['CPUExecutionProvider']
        names = ["X%d" % i for i in range(len(inputs))]
        onx = seq.to_onnx('Y', *names, opset=self.config.opset)
        sess = InferenceSession(onx.SerializeToString(), providers=providers)
        inps = {n: i.astype(np.float32) for n, i in zip(names, inputs)}
        res = sess.run(None, inps)
        return res[0]

    def test_analyse_einsum_equation(self):
        "unit test"
        self.assert_raise(lambda: analyse_einsum_equation("abc"), NotImplementedError)
        self.assert_raise(lambda: analyse_einsum_equation("abc0,ch->ah"), ValueError)
        self.assert_raise(lambda: analyse_einsum_equation("abc,ch->a0"), ValueError)
        res = analyse_einsum_equation("abc,ch->ah")
        self.assertEqual(len(res), 4)
        letters, mat, lengths, duplicates = res
        self.assertEqual(letters, "abch")
        assert_almost_equal(lengths, np.array([3, 2, 2]))
        assert_almost_equal(mat, np.array([[0, 1, 2, -1], [-1, -1, 0, 1], [0, -1, -1, 1]]))
        self.assertEqual(duplicates, [None, None, None])

    def test_analyse_einsum_equation_duplicates(self):
        res = analyse_einsum_equation("aac,ca->aa")
        self.assertEqual(len(res), 4)
        letters, mat, lengths, duplicates = res
        self.assertEqual(letters, "ac")
        assert_almost_equal(lengths, np.array([3, 2, 2]))
        self.assertEqual(duplicates, [{'a': [0, 1], 'c': [2]}, None, {'a': [0, 1]}])
        assert_almost_equal(mat, np.array([[1, 2], [1, 0], [1, -1]]))

    @check_opset_min_version(13, "Squeeze")
    def test_decompose_einsum_equation(self):
        "test decompose einsum"
        m1 = np.arange(0, 8).astype(np.float32).reshape((2, 2, 2))
        m2 = np.arange(0, 4).astype(np.float32).reshape((2, 2))
        exp = np.einsum("bac,ch->ah", m1, m2)
        seq = decompose_einsum_equation("bac,ch->ah", (2, 2, 2), (2, 2))
        dot = seq.to_dot()
        red = dot.split('red')
        self.assertEqual(len(red), 5)
        res = self.apply_einsum_sequence(seq, m1, m2)
        assert_almost_equal(exp, res)

    @check_opset_min_version(13, "Squeeze")
    def test_decompose_einsum_equation_deep_case(self):
        m1 = np.arange(0, 16).astype(np.float32).reshape((2, 2, 2, 2))
        m2 = np.arange(0, 16).astype(np.float32).reshape((2, 2, 2, 2))
        exp = np.einsum("bsnh,btnh->bnts", m1, m2)
        seq = decompose_einsum_equation("bsnh,btnh->bnts")
        res = self.apply_einsum_sequence(seq, m1, m2)
        assert_almost_equal(exp, res)

    @check_opset_min_version(13, "Squeeze")
    def test_decompose_einsum_equation_onnx(self):
        m1 = np.arange(0, 24).astype(np.float32).reshape((2, 3, 4))
        m2 = np.arange(0, 20).astype(np.float32).reshape((4, 5))
        seq = decompose_einsum_equation("bac,ch->ah", (2, 3, 4), (4, 5))
        exp = np.einsum("bac,ch->ah", m1, m2)
        res = self.apply_einsum_sequence(seq, m1, m2)
        assert_almost_equal(exp, res)

    @check_opset_min_version(13, "Squeeze")
    def test_decompose_einsum_equation_noshape(self):
        m1 = np.arange(0, 24).astype(np.float32).reshape((2, 3, 4))
        m2 = np.arange(0, 20).astype(np.float32).reshape((4, 5))
        seq = decompose_einsum_equation("bac,ch->ah")
        exp = np.einsum("bac,ch->ah", m1, m2)
        res = self.apply_einsum_sequence(seq, m1, m2)
        assert_almost_equal(exp, res)

    @check_opset_min_version(13, "Squeeze")
    def test_decompose_einsum_equation_onnx2(self):
        "test bac,cd,def->ebc"
        m1 = np.arange(0, 24).astype(np.float32).reshape((2, 3, 4))
        m2 = np.arange(0, 20).astype(np.float32).reshape((4, 5))
        m3 = np.arange(0, 77 * 5).astype(np.float32).reshape((5, 7, 11))

        seq = decompose_einsum_equation(
            "bac,cd,def->ebc", (2, 3, 4), (4, 5), (5, 7, 11))
        exp = np.einsum("bac,cd,def->ebc", m1, m2, m3)
        res = self.apply_einsum_sequence(seq, m1, m2, m3)
        assert_almost_equal(exp, res)

    def test_einsum_sub_op(self):
        self.assert_raise(lambda: EinsumSubOp(2, "er", (2, 2)), ValueError)
        self.assert_raise(lambda: EinsumSubOp(2, "expand_dims"), RuntimeError)
        self.assert_raise(lambda: EinsumSubOp(2, "matmul", (2, 2)), RuntimeError)
        self.assert_raise(lambda: EinsumSubOp(2, "id", (2, 2)), TypeError)

    def common_test_case_2(self, equation):
        m1 = np.arange(2 * 2 * 2).reshape((2, 2, 2)) + 10
        m2 = np.arange(4).reshape((2, 2)) + 100
        exp = np.einsum(equation, m1, m2)

        seq = decompose_einsum_equation(equation, m1.shape, m2.shape)
        res = self.apply_einsum_sequence(seq, m1, m2)
        assert_almost_equal(exp, res)

    @check_opset_min_version(13, "Squeeze")
    def test_case_2_a(self):
        self.common_test_case_2('abc,cd->abc')

    @check_opset_min_version(13, "Squeeze")
    def test_many_2(self):
        "test many equation with 2 inputs"
        m1 = np.arange(2 * 2 * 2).reshape((2, 2, 2)) + 10
        m2 = np.arange(4).reshape((2, 2)) + 100

        res = []
        for p1 in itertools.permutations(list("abc")):
            for p2 in itertools.permutations(list("cd")):
                for i in [1, 2]:
                    for j in [0, 1]:
                        sp1 = "".join(p1)
                        sp2 = "".join(p2)
                        if len(set([sp1[0], sp1[i], sp2[j]])) != 3:
                            continue
                        equation = "%s,%s->%s%s%s" % (
                            sp1, sp2, sp1[0], sp1[i], sp2[j])
                        try:
                            r = np.einsum(equation, m1, m2)
                            res.append((equation, r))
                        except ValueError:
                            # Not viable equation.
                            continue

        for i, (eq, exp) in enumerate(res):
            with self.subTest(equation=eq, index=i, total=len(res)):
                seq = decompose_einsum_equation(
                    eq, m1.shape, m2.shape)
                res = self.apply_einsum_sequence(seq, m1, m2)
                exp = np.einsum(eq, m1, m2)
                assert_almost_equal(exp, res)

    @check_opset_min_version(13, "Squeeze")
    def test_many_3(self):
        "test many equation with 3 inputs"
        m1 = np.arange(2 * 2 * 2).reshape((2, 2, 2)) + 10
        m2 = np.arange(4).reshape((2, 2)) + 100
        m3 = np.arange(8).reshape((2, 2, 2)) + 1000

        res = []
        for p1 in itertools.permutations(list("abc")):  # pylint: disable=R1702
            for p2 in itertools.permutations(list("cd")):
                for p3 in itertools.permutations(list("def")):
                    for i in [1, 2]:
                        for j in [0, 1]:
                            sp1 = "".join(p1)
                            sp2 = "".join(p2)
                            sp3 = "".join(p3)
                            equation = "%s,%s,%s->%s%s%s" % (
                                sp1, sp2, sp3, sp1[0], sp1[i], sp3[j])
                            try:
                                r = np.einsum(equation, m1, m2, m3)
                                res.append((equation, r))
                            except ValueError:
                                # Not viable equation.
                                continue

        for i, (eq, exp) in enumerate(res):
            with self.subTest(equation=eq, index=i, total=len(res)):
                seq = decompose_einsum_equation(
                    eq, m1.shape, m2.shape, m3.shape)
                res = self.apply_einsum_sequence(seq, m1, m2, m3)
                exp = np.einsum(eq, m1, m2, m3)
                assert_almost_equal(exp, res)

    # Taken from https://github.com/numpy/numpy/blob/main/numpy/
    # core/tests/test_einsum.py.

    def optimize_compare(self, equation, operands=None):
        "Compares numpy einsum and ONNX."
        with self.subTest(equation=equation):
            if operands is not None:
                inputs = operands
            else:
                eqs = equation.split("->")[0].split(",")
                inputs = []
                for d, eq in enumerate(eqs):
                    i = np.arange(2 ** len(eq)).reshape(
                        (2,) * len(eq)).astype(np.float32)
                    inputs.append(
                        i + np.array([3 ** d], dtype=np.float32))

            exp = np.einsum(equation, *inputs)
            shapes = [m.shape for m in inputs]

            seq = decompose_einsum_equation(equation, *shapes)
            got = self.apply_einsum_sequence(seq, *inputs)
            assert_almost_equal(exp, got, decimal=5)

    @check_opset_min_version(13, "Squeeze")
    def test_numpy_test_hadamard_like_products(self):
        self.optimize_compare('a,ab,abc->abc')
        self.optimize_compare('a,b,ab->ab')

    @check_opset_min_version(13, "Squeeze")
    def test_np_test_np_test_collapse(self):
        self.optimize_compare('ab,ab,cd,cd->ac')
        self.optimize_compare('ab,ab,c->c')
        self.optimize_compare('ab,ab,cd,cd->cd')

    @check_opset_min_version(13, "Squeeze")
    def test_np_test_index_transformations(self):
        self.optimize_compare('ea,fb,gc,hd,abcd->efgh')
        self.optimize_compare('ea,fb,abcd,gc,hd->efgh')
        self.optimize_compare('abcd,ea,fb,gc,hd->efgh')

    @check_opset_min_version(13, "Squeeze")
    def test_np_test_expand(self):
        self.optimize_compare('ab,cd,ef->abcdef')
        self.optimize_compare('ab,cd,ef->acdf')
        self.optimize_compare('ab,cd,de->abcde')
        self.optimize_compare('ab,cd,de->be')
        self.optimize_compare('ab,bcd,cd->abcd')
        self.optimize_compare('ab,bcd,cd->abd')

    @check_opset_min_version(13, "Squeeze")
    def test_np_test_edge_cases1(self):
        self.optimize_compare('efc,dbc,acf,fd->abe')
        self.optimize_compare(
            'eac->ace', operands=[np.arange(24).reshape((2, 3, 4))])
        self.optimize_compare('eac->ace')
        self.optimize_compare('bd,db,eac->ace')
        self.optimize_compare('ba,ac,da->bcd')

    @check_opset_min_version(13, "Squeeze")
    def test_np_test_edge_cases2(self):
        self.optimize_compare(
            'eac->ace', operands=[np.arange(24).reshape((2, 3, 4))])
        self.optimize_compare('eb,cb,fb->cef')

    @unittest.skipIf(True, "diagonal still not converted into ONNX")
    def test_np_test_random_cases(self):
        self.optimize_compare('aab,fa,df,ecc->bde')
        self.optimize_compare('bb,ff,be->e')
        self.optimize_compare('afd,ba,cc,dc->bf')
        self.optimize_compare('bbd,bda,fc,db->acf')
        self.optimize_compare('dba,ead,cad->bce')
        self.optimize_compare('aef,fbc,dca->bde')

    def test_np_test_combined_views_mapping(self):
        a = np.arange(9).reshape(1, 1, 3, 1, 3)
        b = np.einsum('bbcdc->d', a)
        assert_almost_equal(b, [12])

    @check_opset_min_version(13, "Squeeze")
    def test_np_test_broadcasting_dot_cases1(self):
        a = np.random.rand(1, 5, 4)
        b = np.random.rand(4, 6)
        c = np.random.rand(5, 6)
        d = np.random.rand(10)
        self.optimize_compare('ijk,kl,jl,i->i', operands=[a, b, c, d])
        e = np.random.rand(1, 1, 5, 4)
        f = np.random.rand(7, 7)
        self.optimize_compare('abjk,kl,jl,ab->ab', operands=[e, b, c, f])

    @check_opset_min_version(13, "Squeeze")
    def test_np_test_broadcasting_dot_cases2(self):
        f = np.arange(7 * 55).reshape(7, 11, 5)
        g = np.arange(30).reshape(2, 3, 5)
        self.optimize_compare('obk,ijk->ioj', operands=[f, g])

    def np_test_complex(self):
        self.optimize_compare('acdf,jbje,gihb,hfac,gfac,gifabc,hfac')
        self.optimize_compare('acdf,jbje,gihb,hfac,gfac,gifabc,hfac')
        self.optimize_compare('cd,bdhe,aidb,hgca,gc,hgibcd,hgac')
        self.optimize_compare('abhe,hidj,jgba,hiab,gab')
        self.optimize_compare('bde,cdh,agdb,hica,ibd,hgicd,hiac')
        self.optimize_compare('chd,bde,agbc,hiad,hgc,hgi,hiad')
        self.optimize_compare('chd,bde,agbc,hiad,bdi,cgh,agdb')
        self.optimize_compare('bdhe,acad,hiab,agac,hibd')

    def np_test_inner_product(self):
        self.optimize_compare('ab,ab')
        self.optimize_compare('ab,ba')
        self.optimize_compare('abc,abc')
        self.optimize_compare('abc,bac')
        self.optimize_compare('abc,cba')

    @unittest.skipIf(True, reason="diagonal still not converted into ONNX")
    def test_np_test_random_cases_difficult(self):
        "unit test"
        self.optimize_compare('db,bc,cfc->d')
        self.optimize_compare('cac,c,h->h')
        self.optimize_compare('cfc,c,h->h')
        self.optimize_compare('cfc,c,d->d')
        self.optimize_compare('c,cfc,d->d')
        self.optimize_compare('d,c,cfc->d')
        self.optimize_compare('d,bc,cfc->d')
        self.optimize_compare('adb,bc,cfc->d')
        self.optimize_compare('adb,bc,fa,cfc->d')
        self.optimize_compare('ecb,fef,bad,ed->ac')
        self.optimize_compare('fdf,cdd,ccd,afe->ae')
        self.optimize_compare('adb,cfc->d')

    @unittest.skipIf(True, "diagonal still not converted into ONNX")
    def test_np_test_edge_cases_duplicate_indices(self):
        self.optimize_compare('dd,fb,be,cdb->cef')
        self.optimize_compare('dcc,fce,ea,dbf->ab')
        self.optimize_compare('ed,fcd,ff,bcf->be')
        self.optimize_compare('baa,dcf,af,cde->be')
        self.optimize_compare('fff,fae,bef,def->abd')

    def test_abbba(self):
        decompose_einsum_equation("ab,b->ba")


if __name__ == "__main__":
    unittest.main()
