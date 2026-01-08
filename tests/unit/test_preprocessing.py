"""
Unit tests for preprocessing module.
"""

import pytest
from src.preprocessing import TicketPreprocessor


class TestTicketPreprocessor:
    """Unit tests for TicketPreprocessor."""
    
    @pytest.fixture
    def preprocessor(self):
        return TicketPreprocessor()
    
    @pytest.fixture
    def preprocessor_no_pii(self):
        return TicketPreprocessor(remove_pii=False)
    
    # Email redaction tests
    def test_email_redaction(self, preprocessor):
        """Test that emails are properly redacted."""
        text = "Contact john.doe@company.com for help"
        result = preprocessor.clean(text)
        assert "[EMAIL]" in result
        assert "john.doe@company.com" not in result
    
    def test_multiple_emails(self, preprocessor):
        """Test multiple email redaction."""
        text = "CC: alice@test.com and bob@example.org"
        result = preprocessor.clean(text)
        assert result.count("[EMAIL]") == 2
    
    def test_email_preservation_when_disabled(self, preprocessor_no_pii):
        """Test that emails are preserved when PII removal is disabled."""
        text = "Email me at test@example.com"
        result = preprocessor_no_pii.clean(text)
        assert "[EMAIL]" not in result
    
    # URL tests
    def test_url_masking(self, preprocessor):
        """Test URL masking."""
        text = "See https://internal.wiki.com/docs for details"
        result = preprocessor.clean(text)
        assert "[URL]" in result
        assert "https://" not in result
    
    def test_www_url_masking(self, preprocessor):
        """Test www URL masking."""
        text = "Visit www.example.com"
        result = preprocessor.clean(text)
        assert "[URL]" in result
    
    # Phone number tests
    def test_phone_redaction(self, preprocessor):
        """Test phone number redaction."""
        text = "Call me at 555-123-4567"
        result = preprocessor.clean(text)
        assert "[PHONE]" in result
        assert "555-123-4567" not in result
    
    # Ticket ID tests
    def test_ticket_id_masking(self, preprocessor):
        """Test ticket ID normalization."""
        text = "Refer to INC0001234 and REQ0005678"
        result = preprocessor.clean(text)
        assert result.count("[TICKET_REF]") == 2
    
    # HTML tests
    def test_html_removal(self, preprocessor):
        """Test HTML tag removal."""
        text = "<p>Hello</p><br/><div>World</div>"
        result = preprocessor.clean(text)
        assert "<" not in result
        assert ">" not in result
        assert "hello" in result
        assert "world" in result
    
    # Edge cases
    def test_empty_input(self, preprocessor):
        """Test empty input handling."""
        assert preprocessor.clean("") == ""
        assert preprocessor.clean(None) == ""
    
    def test_whitespace_normalization(self, preprocessor):
        """Test whitespace normalization."""
        text = "Too   many    spaces\n\nand\tnewlines"
        result = preprocessor.clean(text)
        assert "  " not in result
        assert "\n" not in result
        assert "\t" not in result
    
    def test_lowercasing(self, preprocessor):
        """Test case normalization."""
        text = "UPPERCASE and MixedCase"
        result = preprocessor.clean(text)
        assert result == "uppercase and mixedcase"
    
    def test_lowercasing_disabled(self):
        """Test that lowercasing can be disabled."""
        preprocessor = TicketPreprocessor(lowercase=False)
        text = "UPPERCASE"
        result = preprocessor.clean(text)
        assert result == "UPPERCASE"
    
    # Field combination tests
    def test_field_combination(self, preprocessor):
        """Test subject and description combination."""
        result = preprocessor.combine_fields("VPN Issue", "Cannot connect to network")
        assert "[SUBJECT]" in result
        assert "[DESCRIPTION]" in result
        assert "vpn issue" in result
        assert "cannot connect" in result
    
    def test_field_combination_empty_subject(self, preprocessor):
        """Test combination with empty subject."""
        result = preprocessor.combine_fields("", "Just a description")
        assert "[DESCRIPTION]" in result
        assert "[SUBJECT]" not in result
    
    def test_field_combination_empty_description(self, preprocessor):
        """Test combination with empty description."""
        result = preprocessor.combine_fields("Subject only", "")
        assert "[SUBJECT]" in result
        assert "[DESCRIPTION]" not in result
    
    # Length tests
    def test_max_length_truncation(self):
        """Test text truncation at max length."""
        preprocessor = TicketPreprocessor(max_length=10)
        text = " ".join(["word"] * 20)
        result = preprocessor.clean(text)
        assert len(result.split()) <= 10
    
    # Statistics tests
    def test_text_statistics(self, preprocessor):
        """Test text statistics computation."""
        text = "Hello world test@email.com https://url.com"
        stats = preprocessor.get_text_statistics(text)
        
        assert stats["word_count"] == 4
        assert stats["contains_email"] is True
        assert stats["contains_url"] is True
        assert stats["char_count"] > 0
    
    def test_text_statistics_empty(self, preprocessor):
        """Test statistics for empty text."""
        stats = preprocessor.get_text_statistics("")
        assert stats["word_count"] == 0
        assert stats["char_count"] == 0
