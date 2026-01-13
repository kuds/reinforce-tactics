"""Tests for clipboard utility."""
import sys
import subprocess
from unittest.mock import Mock, patch, MagicMock

import pytest
import pygame

from reinforcetactics.utils.clipboard import get_clipboard_text, set_clipboard_text


@pytest.fixture
def init_pygame():
    """Initialize pygame for tests."""
    if not pygame.get_init():
        pygame.init()
    yield
    # Clean up
    if pygame.get_init():
        pygame.quit()


class TestGetClipboardText:
    """Tests for get_clipboard_text function."""

    def test_pygame_scrap_success(self, init_pygame):
        """Test successful clipboard read using pygame.scrap."""
        test_text = "Test clipboard content"
        
        with patch('pygame.scrap.get_init', return_value=True):
            with patch('pygame.scrap.get', return_value=test_text.encode('utf-8')):
                result = get_clipboard_text()
                assert result == test_text

    def test_pygame_scrap_with_null_chars(self, init_pygame):
        """Test clipboard read strips null characters."""
        test_text = "Test content"
        text_with_nulls = test_text.encode('utf-8') + b'\x00\x00'
        
        with patch('pygame.scrap.get_init', return_value=True):
            with patch('pygame.scrap.get', return_value=text_with_nulls):
                result = get_clipboard_text()
                assert result == test_text

    def test_pygame_scrap_returns_none(self, init_pygame):
        """Test when pygame.scrap returns None."""
        with patch('pygame.scrap.get_init', return_value=True):
            with patch('pygame.scrap.get', return_value=None):
                # Should try macOS fallback if on darwin, otherwise return None
                if sys.platform == 'darwin':
                    with patch('subprocess.run') as mock_run:
                        mock_run.return_value = Mock(returncode=1, stdout='')
                        result = get_clipboard_text()
                        assert result is None
                else:
                    result = get_clipboard_text()
                    assert result is None

    def test_pygame_scrap_not_initialized(self, init_pygame):
        """Test when pygame.scrap is not initialized."""
        with patch('pygame.scrap.get_init', return_value=False):
            # Should try macOS fallback if on darwin, otherwise return None
            if sys.platform == 'darwin':
                with patch('subprocess.run') as mock_run:
                    mock_run.return_value = Mock(returncode=1, stdout='')
                    result = get_clipboard_text()
                    assert result is None
            else:
                result = get_clipboard_text()
                assert result is None

    def test_pygame_scrap_error(self, init_pygame):
        """Test when pygame.scrap raises an error."""
        with patch('pygame.scrap.get_init', return_value=True):
            with patch('pygame.scrap.get', side_effect=pygame.error("Clipboard error")):
                # Should try macOS fallback if on darwin, otherwise return None
                if sys.platform == 'darwin':
                    with patch('subprocess.run') as mock_run:
                        mock_run.return_value = Mock(returncode=1, stdout='')
                        result = get_clipboard_text()
                        assert result is None
                else:
                    result = get_clipboard_text()
                    assert result is None

    def test_unicode_decode_error(self, init_pygame):
        """Test handling of invalid UTF-8 data."""
        invalid_utf8 = b'\x80\x81\x82'
        
        with patch('pygame.scrap.get_init', return_value=True):
            with patch('pygame.scrap.get', return_value=invalid_utf8):
                # Should try macOS fallback if on darwin, otherwise return None
                if sys.platform == 'darwin':
                    with patch('subprocess.run') as mock_run:
                        mock_run.return_value = Mock(returncode=1, stdout='')
                        result = get_clipboard_text()
                        assert result is None
                else:
                    result = get_clipboard_text()
                    assert result is None

    @pytest.mark.skipif(sys.platform != 'darwin', reason="macOS-specific test")
    def test_macos_pbpaste_success(self, init_pygame):
        """Test successful clipboard read using pbpaste on macOS."""
        test_text = "Test clipboard content"
        
        with patch('pygame.scrap.get_init', return_value=False):
            with patch('subprocess.run') as mock_run:
                mock_run.return_value = Mock(returncode=0, stdout=test_text)
                result = get_clipboard_text()
                assert result == test_text
                mock_run.assert_called_once_with(
                    ['pbpaste'],
                    capture_output=True,
                    text=True,
                    timeout=1
                )

    @pytest.mark.skipif(sys.platform != 'darwin', reason="macOS-specific test")
    def test_macos_pbpaste_failure(self, init_pygame):
        """Test when pbpaste fails on macOS."""
        with patch('pygame.scrap.get_init', return_value=False):
            with patch('subprocess.run') as mock_run:
                mock_run.return_value = Mock(returncode=1, stdout='')
                result = get_clipboard_text()
                assert result is None

    @pytest.mark.skipif(sys.platform != 'darwin', reason="macOS-specific test")
    def test_macos_pbpaste_not_found(self, init_pygame):
        """Test when pbpaste is not found on macOS."""
        with patch('pygame.scrap.get_init', return_value=False):
            with patch('subprocess.run', side_effect=FileNotFoundError()):
                result = get_clipboard_text()
                assert result is None

    @pytest.mark.skipif(sys.platform != 'darwin', reason="macOS-specific test")
    def test_macos_pbpaste_timeout(self, init_pygame):
        """Test when pbpaste times out on macOS."""
        with patch('pygame.scrap.get_init', return_value=False):
            with patch('subprocess.run', side_effect=subprocess.TimeoutExpired('pbpaste', 1)):
                result = get_clipboard_text()
                assert result is None


class TestSetClipboardText:
    """Tests for set_clipboard_text function."""

    def test_pygame_scrap_success(self, init_pygame):
        """Test successful clipboard write using pygame.scrap."""
        test_text = "Test clipboard content"
        
        with patch('pygame.scrap.get_init', return_value=True):
            with patch('pygame.scrap.put') as mock_put:
                result = set_clipboard_text(test_text)
                assert result is True
                mock_put.assert_called_once_with(pygame.SCRAP_TEXT, test_text.encode('utf-8'))

    def test_pygame_scrap_not_initialized(self, init_pygame):
        """Test when pygame.scrap is not initialized."""
        with patch('pygame.scrap.get_init', return_value=False):
            # Should try macOS fallback if on darwin, otherwise return False
            if sys.platform == 'darwin':
                with patch('subprocess.run') as mock_run:
                    mock_run.return_value = Mock(returncode=1)
                    result = set_clipboard_text("test")
                    assert result is False
            else:
                result = set_clipboard_text("test")
                assert result is False

    def test_pygame_scrap_error(self, init_pygame):
        """Test when pygame.scrap raises an error."""
        test_text = "Test clipboard content"
        
        with patch('pygame.scrap.get_init', return_value=True):
            with patch('pygame.scrap.put', side_effect=pygame.error("Clipboard error")):
                # Should try macOS fallback if on darwin, otherwise return False
                if sys.platform == 'darwin':
                    with patch('subprocess.run') as mock_run:
                        mock_run.return_value = Mock(returncode=1)
                        result = set_clipboard_text(test_text)
                        assert result is False
                else:
                    result = set_clipboard_text(test_text)
                    assert result is False

    @pytest.mark.skipif(sys.platform != 'darwin', reason="macOS-specific test")
    def test_macos_pbcopy_success(self, init_pygame):
        """Test successful clipboard write using pbcopy on macOS."""
        test_text = "Test clipboard content"
        
        with patch('pygame.scrap.get_init', return_value=False):
            with patch('subprocess.run') as mock_run:
                result = set_clipboard_text(test_text)
                assert result is True
                mock_run.assert_called_once_with(
                    ['pbcopy'],
                    input=test_text.encode('utf-8'),
                    timeout=1,
                    check=True
                )

    @pytest.mark.skipif(sys.platform != 'darwin', reason="macOS-specific test")
    def test_macos_pbcopy_failure(self, init_pygame):
        """Test when pbcopy fails on macOS."""
        with patch('pygame.scrap.get_init', return_value=False):
            with patch('subprocess.run', side_effect=subprocess.CalledProcessError(1, 'pbcopy')):
                result = set_clipboard_text("test")
                assert result is False

    @pytest.mark.skipif(sys.platform != 'darwin', reason="macOS-specific test")
    def test_macos_pbcopy_not_found(self, init_pygame):
        """Test when pbcopy is not found on macOS."""
        with patch('pygame.scrap.get_init', return_value=False):
            with patch('subprocess.run', side_effect=FileNotFoundError()):
                result = set_clipboard_text("test")
                assert result is False

    @pytest.mark.skipif(sys.platform != 'darwin', reason="macOS-specific test")
    def test_macos_pbcopy_timeout(self, init_pygame):
        """Test when pbcopy times out on macOS."""
        with patch('pygame.scrap.get_init', return_value=False):
            with patch('subprocess.run', side_effect=subprocess.TimeoutExpired('pbcopy', 1)):
                result = set_clipboard_text("test")
                assert result is False
