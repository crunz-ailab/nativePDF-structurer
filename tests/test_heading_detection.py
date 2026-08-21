import unittest
#@GPT 星号导入导致Pyflakes无法验证测试符号；改为显式列出实际使用的四个谓词。
# from nativePDF_structurer.heading_detection import *
from nativePDF_structurer.heading_detection import (
    looks_like_heading_text,
    looks_like_numeric_or_rule_label,
    looks_like_short_colon_label,
    looks_like_toc_entry,
)

class HeadingPredicateTests(unittest.TestCase):
    def test_generic_predicates(self):
        self.assertTrue(looks_like_toc_entry("标题 ...... 12"))
        self.assertTrue(looks_like_numeric_or_rule_label("── 3"))
        self.assertTrue(looks_like_short_colon_label("注："))
        self.assertTrue(looks_like_heading_text("1.2 产品概述"))
        self.assertFalse(looks_like_heading_text("4 ───"))

if __name__ == "__main__": unittest.main()
