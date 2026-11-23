"""前端结构回归：防止三线合并类事故再次留在页面里。

2026-11-23 发现 index.html 残留一行 ``=======`` 及其后约 50 行旧版 DOM，
与新版形成重复 ID（create-form、view-projects、music-modal 等），
页面行为不可预测。本文件用静态检查守住两条底线：

1. 不得残留任何 git 合并冲突标记；
2. 同一 HTML 文档内 id 不得重复。
"""

import re
import unittest
from pathlib import Path

from backend.app.config import PROJECT_ROOT

FRONTEND_FILES = ("index.html",)


class FrontendStructureTests(unittest.TestCase):
    def _read(self, name: str) -> str:
        return (PROJECT_ROOT / "frontend" / name).read_text(encoding="utf-8")

    def test_no_conflict_markers(self):
        for name in FRONTEND_FILES:
            text = self._read(name)
            for marker in ("<<<<<<<", "=======", ">>>>>>>"):
                self.assertNotIn(
                    marker,
                    text,
                    msg=f"{name} 残留合并冲突标记 {marker}；若是有意输出请换写法",
                )

    def test_no_duplicate_ids(self):
        for name in FRONTEND_FILES:
            text = self._read(name)
            ids = re.findall(r'\bid="([^"]+)"', text)
            duplicated = sorted({value for value in ids if ids.count(value) > 1})
            self.assertEqual(
                duplicated,
                [],
                msg=f"{name} 存在重复 id：{duplicated}（会导致 DOM 行为不可预测）",
            )

    def test_scripts_mounted_once(self):
        text = self._read("index.html")
        self.assertEqual(text.count('src="/app.js"'), 1)
        self.assertEqual(text.count('src="/voice-studio.js"'), 1)
        self.assertEqual(text.count('src="/wizard.js"'), 1)
        self.assertTrue(text.rstrip().endswith("</html>"))

    def test_wizard_defines_five_steps_and_loads_before_app(self):
        wizard = self._read("wizard.js")
        for label in ("需求", "选镜头", "审核脚本", "审核 prompt", "生成验收"):
            self.assertIn(label, wizard, f"wizard.js 缺少步骤 {label}")
        index = self._read("index.html")
        self.assertLess(
            index.index('src="/wizard.js"'), index.index('src="/app.js"'),
            "wizard.js 必须先于 app.js 加载",
        )


if __name__ == "__main__":
    unittest.main()
