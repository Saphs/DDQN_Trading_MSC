import unittest

from pandas import Series

from metrics import calc


class TestMetrics(unittest.TestCase):

    def test_arithmetic_mean(self):
        values = Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = calc.arithmetic_mean(values)
        self.assertEqual(result, 3.0)

    def test_geometric_mean(self):
        values = Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = calc.geometric_mean(values)
        self.assertEqual(result, 2.6051710846973517)


if __name__ == '__main__':
    unittest.main()