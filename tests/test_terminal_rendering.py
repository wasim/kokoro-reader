import io
import threading
import unittest
from unittest.mock import patch

from aloud import Player, Segment


class TerminalRenderingTests(unittest.TestCase):
    def test_commit_returns_to_first_column_before_clearing_live_region(self):
        player = Player.__new__(Player)
        player.chunks = [Segment("one two three four five six")]
        player.out_lock = threading.Lock()
        player._above = 2
        player._para_lines = 2
        output = io.StringIO()

        with patch.object(Player, "_term_width", return_value=18), \
                patch("sys.stdout", output):
            player._commit(0)

        self.assertEqual(
            output.getvalue(),
            "\x1b[2A\r\x1b[Jone two three four\nfive six\n",
        )


if __name__ == "__main__":
    unittest.main()
