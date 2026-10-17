"""End-to-end desktop controls; CI runs this under Xvfb on Linux."""
import os
import time
import unittest

import numpy as np


@unittest.skipUnless(os.environ.get('DISPLAY'), 'Needs an X display; run with xvfb-run -a')
class DesktopTests(unittest.TestCase):
    def test_teach_probe_clear_and_erase_use_the_live_medium(self):
        import tkinter as tk
        from solsbaby.gui import Laboratory
        root = tk.Tk()
        try:
            root.geometry('1200x850')
            app = Laboratory(root, seed=3)
            root.update()
            def finish_animation():
                deadline = time.monotonic()+30
                while app.busy and time.monotonic() < deadline:
                    root.update()
                    time.sleep(.005)
                self.assertFalse(app.busy, 'GUI animation did not finish')
                root.update()
            app.teach(0)
            finish_animation()
            self.assertGreater(app.state.tag.max(), .2)
            learned = app.state.tag.copy()
            app.probe()
            finish_animation()
            np.testing.assert_array_equal(app.state.tag, learned)
            self.assertIsNotNone(app.last_measurement)
            app.pulse()
            finish_animation()
            self.assertGreater(np.abs(app.state.v).max(), .01)
            retained = app.state.tag.copy()
            app.clear_fast()
            np.testing.assert_array_equal(app.state.v, 0)
            np.testing.assert_array_equal(app.state.tag, retained)
            app.erase()
            np.testing.assert_array_equal(app.state.tag, 0)
            app.active.set(True)
            app.switch_active()
            self.assertEqual(app.medium.parameters.active_strength, 1.2)
            app.seed.set(7)
            app.regrow()
            self.assertEqual(app.current_seed, 7)
            np.testing.assert_array_equal(app.state.tag, 0)
        finally:
            root.destroy()


if __name__ == '__main__':
    unittest.main()
