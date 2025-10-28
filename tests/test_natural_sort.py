"""Unit tests for natural_sort function."""

from omm_fts.utils.natural_sort import natural_sort


class TestNaturalSort:
    """Test suite for natural_sort function."""

    def test_simple_numeric_sort(self):
        """Test sorting strings with numbers."""
        items = ["file10.txt", "file2.txt", "file1.txt", "file20.txt"]
        expected = ["file1.txt", "file2.txt", "file10.txt", "file20.txt"]
        assert natural_sort(items) == expected

    def test_alphabetic_sort(self):
        """Test sorting purely alphabetic strings."""
        items = ["charlie", "alpha", "bravo"]
        expected = ["alpha", "bravo", "charlie"]
        assert natural_sort(items) == expected

    def test_mixed_case_sort(self):
        """Test case-insensitive sorting."""
        items = ["File10.txt", "file2.txt", "FILE1.txt"]
        expected = ["FILE1.txt", "file2.txt", "File10.txt"]
        assert natural_sort(items) == expected

    def test_multiple_numbers_in_string(self):
        """Test sorting with multiple numbers in strings."""
        items = ["a10b20", "a2b30", "a10b5", "a2b20"]
        expected = ["a2b20", "a2b30", "a10b5", "a10b20"]
        assert natural_sort(items) == expected

    def test_empty_list(self):
        """Test sorting empty list."""
        assert natural_sort([]) == []

    def test_single_element(self):
        """Test sorting single element list."""
        assert natural_sort(["file1.txt"]) == ["file1.txt"]

    def test_no_numbers(self):
        """Test sorting strings without numbers."""
        items = ["zebra", "apple", "mango"]
        expected = ["apple", "mango", "zebra"]
        assert natural_sort(items) == expected

    def test_leading_zeros(self):
        """Test handling of leading zeros."""
        items = ["file001.txt", "file10.txt", "file002.txt"]
        expected = ["file001.txt", "file002.txt", "file10.txt"]
        assert natural_sort(items) == expected

    def test_negative_numbers(self):
        """Test handling of negative numbers (treated as text)."""
        items = ["file-10.txt", "file-2.txt", "file-1.txt"]
        # The minus sign is treated as text, so it sorts alphabetically
        result = natural_sort(items)
        assert len(result) == 3  # noqa: PLR2004
        assert "file-1.txt" in result

    def test_special_characters(self):
        """Test sorting with special characters."""
        items = ["file_10.txt", "file-2.txt", "file.1.txt"]
        result = natural_sort(items)
        assert len(result) == 3  # noqa: PLR2004

    def test_paths(self):
        """Test sorting file paths."""
        items = [
            "/path/to/file10.txt",
            "/path/to/file2.txt",
            "/path/to/file1.txt",
        ]
        expected = [
            "/path/to/file1.txt",
            "/path/to/file2.txt",
            "/path/to/file10.txt",
        ]
        assert natural_sort(items) == expected
