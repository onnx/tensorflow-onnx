"""
@brief      test log(time=6s)
"""
import unittest
import io
from contextlib import redirect_stdout
import itertools
import numpy as np
from numpy.testing import assert_almost_equal
from tf2onnx.optimizer.einsum_optimizer import (
    numpy_diagonal, numpy_extended_dot,
    analyse_einsum_equation, decompose_einsum_equation, EinsumSubOp,
    apply_einsum_sequence)


class TestEinsum(unittest.TestCase):

    def assertRaise(self, fct, exc_type):
        try:
            fct()
        except exc_type as e:
            return
        raise AssertionError("%r was not raised." % exc_type)

    def test_numpy_diagonal(self):
        mat = np.arange(8).reshape((2, 2, 2))
        diag = numpy_diagonal(mat, 1, [1, 2])
        assert_almost_equal(diag, np.array([[0, 3], [4, 7]]))
        diag = numpy_diagonal(mat, 2, [1, 2])
        assert_almost_equal(diag, np.array([[0, 3], [4, 7]]))

        diag = numpy_diagonal(mat, 0, [0, 1])
        assert_almost_equal(diag, np.array([[0, 1], [6, 7]]))
        diag = numpy_diagonal(mat, 1, [0, 1])
        assert_almost_equal(diag, np.array([[0, 1], [6, 7]]))

        diag = numpy_diagonal(mat, 0, [0, 2])
        assert_almost_equal(diag, np.array([[0, 2], [5, 7]]))
        diag = numpy_diagonal(mat, 2, [0, 2])
        assert_almost_equal(diag, np.array([[0, 2], [5, 7]]).T)

    def test_numpy_extended_dot_2_a(self):
        m1 = np.arange(4).reshape((2, 2)).astype(np.float32)
        m2 = m1 + 10

        self.assertRaise(lambda: numpy_extended_dot(m1, m2.T, [0], [1], [2]),
                         ValueError)
        dm1 = m1.reshape((2, 2, 1))
        dm2 = m2.reshape((1, 2, 2))
        dot = numpy_extended_dot(dm1, dm2, axes=[1], left=[0], right=[2])
        exp = m1 @ m2
        assert_almost_equal(exp, np.squeeze(dot))

        dm1 = m1.reshape((2, 1, 2))
        dm2 = m2.reshape((1, 2, 2))
        dot = numpy_extended_dot(dm1, dm2, axes=[2], left=[0], right=[1])
        exp = m1 @ m2.T
        assert_almost_equal(exp, np.squeeze(dot))

    def test_numpy_extended_dot_3(self):
        m1 = np.arange(8).reshape((2, 2, 2))
        m2 = m1 + 10

        dot = numpy_extended_dot(m1, m2, [1], [0], [2])
        exp = np.array([[[164, 176]], [[580, 624]]])
        assert_almost_equal(exp, dot)

        dot = numpy_extended_dot(m1, m2, [1], [2], [0])
        exp = np.array([[[284, 376]], [[380, 504]]])
        assert_almost_equal(exp, dot)

        dot = numpy_extended_dot(m1, m2, [1], [2], [0, 1])
        exp = np.array([[[84, 126], [200, 250]],
                           [[116, 174], [264, 330]]])
        assert_almost_equal(exp, dot)

    def test_analyse_einsum_equation(self):
        self.assertRaise(lambda: analyse_einsum_equation("abc"),
                         NotImplementedError)
        self.assertRaise(lambda: analyse_einsum_equation("abc0,ch->ah"),
                         ValueError)
        self.assertRaise(lambda: analyse_einsum_equation("abc,ch->a0"),
                         ValueError)
        res = analyse_einsum_equation("abc,ch->ah")
        self.assertEqual(len(res), 4)
        letters, mat, lengths, duplicates = res
        self.assertEqual(letters, "abch")
        assert_almost_equal(lengths, np.array([3, 2, 2]))
        assert_almost_equal(
            mat, np.array([[0, 1, 2, -1],
                              [-1, -1, 0, 1],
                              [0, -1, -1, 1]]))
        self.assertEqual(duplicates, [None, None, None])

    def test_analyse_einsum_equation_duplicates(self):
        res = analyse_einsum_equation("aac,ca->aa")
        self.assertEqual(len(res), 4)
        letters, mat, lengths, duplicates = res
        self.assertEqual(letters, "ac")
        assert_almost_equal(lengths, np.array([3, 2, 2]))
        self.assertEqual(duplicates, [{'a': [0, 1], 'c': [2]},
                                      None,
                                      {'a': [0, 1]}])
        assert_almost_equal(
            mat, np.array([[1, 2],
                              [1, 0],
                              [1, -1]]))

    def test_decompose_einsum_equation_exc(self):
        self.assertRaise(
            lambda: decompose_einsum_equation("abc,ch->ah", (2, 2, 2), (2, 2),
                                              strategy="donotexist"),
            ValueError)
        self.assertRaise(
            lambda: decompose_einsum_equation("abc,ch->ah"), ValueError)
        self.assertRaise(
            lambda: decompose_einsum_equation("abc,ch->ah", (2, 2, 2), (2, 2),
                                              "donotexist"),
            TypeError)
        self.assertRaise(
            lambda: decompose_einsum_equation("abc,ch->ah", (2, 2, 2)),
            ValueError)
        self.assertRaise(
            lambda: decompose_einsum_equation("abc,ch->ah", (2, 2), (2, 2)),
            ValueError)

    def test_decompose_einsum_equation(self):
        m1 = np.arange(0, 8).astype(np.float32).reshape((2, 2, 2))
        m2 = np.arange(0, 4).astype(np.float32).reshape((2, 2))
        exp = np.einsum("bac,ch->ah", m1, m2)

        def fct():
            print("########################## DECOMPOSE")
            seq = decompose_einsum_equation("bac,ch->ah", (2, 2, 2), (2, 2))
            print("########################## APPLY")
            dot = seq.to_dot()
            print(dot)
            red = dot.split('red')
            self.assertEqual(len(red), 4)
            res = apply_einsum_sequence(seq, m1, m2)
            print("########################## END")
            return res

        f = io.StringIO()
        try:
            with redirect_stdout(f):
                res = fct()
        except Exception as e:
            raise AssertionError("Issue. Logs =\n%s" % f.getvalue()) from e

        out = f.getvalue()
        self.assertIn("DECOMPOSE", out)
        assert_almost_equal(exp, res)

    def test_einsum_sub_op(self):
        self.assertRaise(lambda: EinsumSubOp(2, "er", (2, 2)), ValueError)
        self.assertRaise(lambda: EinsumSubOp(2, "expand_dims"), RuntimeError)
        self.assertRaise(lambda: EinsumSubOp(
            2, "matmul", (2, 2)), RuntimeError)
        self.assertRaise(lambda: EinsumSubOp(2, "id", (2, 2)), TypeError)

    def test_case_1_iii_ii_i(self):
        equation = 'ii->i'
        m1 = np.arange(2 * 2).reshape((2, 2)) + 10
        exp = np.einsum(equation, m1)
        seq = decompose_einsum_equation(
            equation, m1.shape)
        res = apply_einsum_sequence(seq, m1)
        assert_almost_equal(exp, res)

    def test_case_1_iii_ii_i_j(self):
        equation = 'iij->ij'
        m1 = np.arange(2 * 2 * 2).reshape((2, 2, 2)) + 10
        exp = np.einsum(equation, m1)
        seq = decompose_einsum_equation(
            equation, m1.shape)
        dot = seq.to_dot()
        self.assertIn("i=0,1", dot)
        res = apply_einsum_sequence(seq, m1)
        assert_almost_equal(exp, res)

    def common_test_case_2(self, equation):
        m1 = np.arange(2 * 2 * 2).reshape((2, 2, 2)) + 10
        m2 = np.arange(4).reshape((2, 2)) + 100
        exp = np.einsum(equation, m1, m2)
        seq = decompose_einsum_equation(
            equation, m1.shape, m2.shape)
        res = apply_einsum_sequence(seq, m1, m2)
        assert_almost_equal(exp, res)

    def test_case_2_A(self):
        self.common_test_case_2('abc,cd->abc')

    def test_many_2(self):
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
                res = apply_einsum_sequence(seq, m1, m2)
                assert_almost_equal(exp, res)

    def test_many_3(self):
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
                res = apply_einsum_sequence(seq, m1, m2, m3)
                assert_almost_equal(exp, res)

    # Taken from https://github.com/numpy/numpy/blob/main/numpy/
    # core/tests/test_einsum.py.

    def optimize_compare(self, equation, operands=None):
        if operands is not None:
            inputs = operands
        else:
            eqs = equation.split("->")[0].split(",")
            inputs = []
            for eq in eqs:
                i = np.arange(2 ** len(eq)).reshape(
                    (2,) * len(eq)).astype(np.float32)
                inputs.append(i + np.array([10], dtype=np.float32))

        exp = np.einsum(equation, *inputs)
        shapes = [m.shape for m in inputs]
        seq = decompose_einsum_equation(equation, *shapes)
        got = apply_einsum_sequence(seq, *inputs)
        assert_almost_equal(exp, got)

    def test_numpy_test_hadamard_like_products(self):
        # Hadamard outer products
        self.optimize_compare('a,ab,abc->abc')
        self.optimize_compare('a,b,ab->ab')

    def test_np_test_np_test_collapse(self):
        # Inner products
        self.optimize_compare('ab,ab,c->c')
        self.optimize_compare('ab,ab,cd,cd->ac')
        self.optimize_compare('ab,ab,cd,cd->cd')
        # self.optimize_compare('ab,ab,c->')
        # self.optimize_compare('ab,ab,cd,cd->')
        # self.optimize_compare('ab,ab,cd,cd,ef,ef->')

    def test_np_test_index_transformations(self):
        # Simple index transformation cases
        self.optimize_compare('ea,fb,gc,hd,abcd->efgh')
        self.optimize_compare('ea,fb,abcd,gc,hd->efgh')
        self.optimize_compare('abcd,ea,fb,gc,hd->efgh')

    def test_np_test_expand(self):
        # Outer products
        self.optimize_compare('ab,cd,ef->abcdef')
        self.optimize_compare('ab,cd,ef->acdf')
        self.optimize_compare('ab,cd,de->abcde')
        self.optimize_compare('ab,cd,de->be')
        self.optimize_compare('ab,bcd,cd->abcd')
        self.optimize_compare('ab,bcd,cd->abd')

    def test_np_test_edge_cases(self):
        # Difficult edge cases for optimization
        self.optimize_compare(
            'eac->ace', operands=[np.arange(24).reshape((2, 3, 4))])
        self.optimize_compare('eac->ace')
        self.optimize_compare('bd,db,eac->ace')
        self.optimize_compare('eb,cb,fb->cef')
        self.optimize_compare('efc,dbc,acf,fd->abe')
        self.optimize_compare('ba,ac,da->bcd')

    def test_np_test_random_cases(self):
        # Randomly built test cases
        self.optimize_compare('aab,fa,df,ecc->bde')
        self.optimize_compare('bb,ff,be->e')
        self.optimize_compare('afd,ba,cc,dc->bf')
        self.optimize_compare('bbd,bda,fc,db->acf')
        self.optimize_compare('dba,ead,cad->bce')
        self.optimize_compare('aef,fbc,dca->bde')

    def test_np_test_combined_views_mapping(self):
        # gh-10792
        a = np.arange(9).reshape(1, 1, 3, 1, 3)
        b = np.einsum('bbcdc->d', a)
        assert_almost_equal(b, [12])

    def test_np_test_broadcasting_dot_cases1(self):
        # Ensures broadcasting cases are not mistaken for GEMM

        a = np.random.rand(1, 5, 4)
        b = np.random.rand(4, 6)
        c = np.random.rand(5, 6)
        d = np.random.rand(10)

        # self.optimize_compare('ijk,kl,jl', operands=[a, b, c])
        self.optimize_compare('ijk,kl,jl,i->i', operands=[a, b, c, d])

        e = np.random.rand(1, 1, 5, 4)
        f = np.random.rand(7, 7)
        # self.optimize_compare('abjk,kl,jl', operands=[e, b, c])
        self.optimize_compare('abjk,kl,jl,ab->ab', operands=[e, b, c, f])

    def test_np_test_broadcasting_dot_cases2(self):
        # Edge case found in gh-11308
        g = np.arange(64).reshape(2, 4, 8)
        self.optimize_compare('obk,ijk->ioj', operands=[g, g])

    def np_test_complex(self):
        # Long test cases
        self.optimize_compare('acdf,jbje,gihb,hfac,gfac,gifabc,hfac')
        self.optimize_compare('acdf,jbje,gihb,hfac,gfac,gifabc,hfac')
        self.optimize_compare('cd,bdhe,aidb,hgca,gc,hgibcd,hgac')
        self.optimize_compare('abhe,hidj,jgba,hiab,gab')
        self.optimize_compare('bde,cdh,agdb,hica,ibd,hgicd,hiac')
        self.optimize_compare('chd,bde,agbc,hiad,hgc,hgi,hiad')
        self.optimize_compare('chd,bde,agbc,hiad,bdi,cgh,agdb')
        self.optimize_compare('bdhe,acad,hiab,agac,hibd')

    def np_test_inner_product(self):
        # Inner products
        self.optimize_compare('ab,ab')
        self.optimize_compare('ab,ba')
        self.optimize_compare('abc,abc')
        self.optimize_compare('abc,bac')
        self.optimize_compare('abc,cba')

    def test_np_test_random_cases_difficult(self):
        self.optimize_compare('cac,c,h->h')
        self.optimize_compare('cfc,c,h->h')
        self.optimize_compare('cfc,c,d->d')
        self.optimize_compare('c,cfc,d->d')
        self.optimize_compare('d,c,cfc->d')
        self.optimize_compare('d,bc,cfc->d')
        self.optimize_compare('db,bc,cfc->d')
        self.optimize_compare('adb,bc,cfc->d')
        self.optimize_compare('adb,bc,fa,cfc->d')
        self.optimize_compare('ecb,fef,bad,ed->ac')
        self.optimize_compare('fdf,cdd,ccd,afe->ae')
        self.optimize_compare('adb,cfc->d')

    def test_np_test_edge_cases_duplicate_indices(self):
        # Difficult edge cases for optimization
        # self.optimize_compare('bca,cdb,dbf,afc->')
        self.optimize_compare('dd,fb,be,cdb->cef')
        self.optimize_compare('dcc,fce,ea,dbf->ab')
        # self.optimize_compare('abcd,ad')
        self.optimize_compare('ed,fcd,ff,bcf->be')
        self.optimize_compare('baa,dcf,af,cde->be')
        self.optimize_compare('fff,fae,bef,def->abd')


if __name__ == "__main__":
    unittest.main()
