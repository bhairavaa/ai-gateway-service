from app.providers.common import normalize_content


class TestNormalizeContent:
    def test_plain_string_passes_through(self):
        assert normalize_content("hello") == "hello"

    def test_single_text_block_extracted(self):
        content = [{"type": "text", "text": "hello"}]
        assert normalize_content(content) == "hello"

    def test_multiple_text_blocks_concatenated(self):
        content = [{"type": "text", "text": "hel"}, {"type": "text", "text": "lo"}]
        assert normalize_content(content) == "hello"

    def test_non_text_blocks_are_dropped(self):
        # Real shape seen in production: Anthropic mixing a citation/tool block with text.
        content = [
            {"type": "text", "text": "The answer is "},
            {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {}},
            {"type": "text", "text": "42."},
        ]
        assert normalize_content(content) == "The answer is 42."

    def test_plain_string_items_in_list_are_kept(self):
        assert normalize_content(["hel", "lo"]) == "hello"

    def test_empty_list_returns_empty_string(self):
        assert normalize_content([]) == ""

    def test_block_with_missing_text_key_treated_as_empty(self):
        assert normalize_content([{"type": "text"}]) == ""
