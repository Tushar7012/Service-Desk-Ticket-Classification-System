"""
Production-grade text preprocessing for service desk tickets.

Design Philosophy:
- Google: Consistent preprocessing ensures reproducibility and prevents subtle bugs
- Amazon: PII redaction is mandatory for customer trust, processing speed matters
"""

import re
import unicodedata
from typing import Optional


class TicketPreprocessor:
    """
    Production-grade text preprocessing for service desk tickets.
    
    Design Decisions:
    - Google: Consistent preprocessing ensures reproducibility and prevents
              subtle bugs in evaluation pipelines.
    - Amazon: PII redaction is mandatory for customer trust. Processing
              speed matters for real-time inference.
    
    Example:
        >>> preprocessor = TicketPreprocessor()
        >>> text = "Contact john@company.com for help with https://issue.com"
        >>> preprocessor.clean(text)
        'contact [EMAIL] for help with [URL]'
    """
    
    def __init__(
        self,
        max_length: int = 512,
        lowercase: bool = True,
        remove_pii: bool = True,
        handle_multilingual: bool = True
    ):
        """
        Initialize the preprocessor.
        
        Args:
            max_length: Maximum number of words to keep
            lowercase: Whether to lowercase text
            remove_pii: Whether to redact PII (emails, phones)
            handle_multilingual: Whether to normalize unicode
        """
        self.max_length = max_length
        self.lowercase = lowercase
        self.remove_pii = remove_pii
        self.handle_multilingual = handle_multilingual
        
        # Compiled patterns for performance (Amazon: latency matters)
        self._email_pattern = re.compile(r'\b[\w.-]+@[\w.-]+\.\w+\b')
        self._phone_pattern = re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b')
        self._url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self._ticket_id_pattern = re.compile(r'INC\d{7}|REQ\d{7}|CHG\d{7}|SR\d{7}')
        self._html_pattern = re.compile(r'<[^>]+>')
        self._whitespace_pattern = re.compile(r'\s+')
        
        # Common noise patterns in tickets
        self._signature_pattern = re.compile(
            r'(regards|thanks|thank you|best|sincerely|cheers)[\s,]*[\w\s]*$',
            re.IGNORECASE
        )
        self._forwarded_pattern = re.compile(
            r'(-{3,}|_{3,})?\s*(forwarded|original message|from:|sent:|to:|subject:)',
            re.IGNORECASE
        )
    
    def clean(self, text: str) -> str:
        """
        Apply all cleaning steps to input text.
        
        Args:
            text: Raw ticket text
            
        Returns:
            Cleaned and normalized text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Remove HTML tags
        text = self._html_pattern.sub(' ', text)
        
        # Mask URLs (preserve semantic: "see link" is useful context)
        text = self._url_pattern.sub('[URL]', text)
        
        # PII Redaction (Amazon: customer trust is paramount)
        if self.remove_pii:
            text = self._email_pattern.sub('[EMAIL]', text)
            text = self._phone_pattern.sub('[PHONE]', text)
        
        # Normalize ticket references
        text = self._ticket_id_pattern.sub('[TICKET_REF]', text)
        
        # Remove forwarded message artifacts
        text = self._forwarded_pattern.sub(' ', text)
        
        # Remove email signatures (often noise)
        text = self._signature_pattern.sub('', text)
        
        # Unicode normalization (Google: consistent representation)
        if self.handle_multilingual:
            text = unicodedata.normalize('NFKC', text)
        
        # Case normalization
        if self.lowercase:
            text = text.lower()
        
        # Whitespace normalization
        text = self._whitespace_pattern.sub(' ', text).strip()
        
        # Length truncation (by words)
        words = text.split()
        if len(words) > self.max_length:
            text = ' '.join(words[:self.max_length])
        
        return text
    
    def combine_fields(
        self,
        subject: Optional[str],
        description: Optional[str],
        separator: str = " [SEP] "
    ) -> str:
        """
        Combine subject and description with special tokens.
        
        Subject is high-signal, so we prefix it distinctly.
        
        Args:
            subject: Ticket subject line
            description: Ticket body/description
            separator: Separator token between fields
            
        Returns:
            Combined and cleaned text
        """
        subject_clean = self.clean(subject or "")
        desc_clean = self.clean(description or "")
        
        if subject_clean and desc_clean:
            return f"[SUBJECT] {subject_clean}{separator}[DESCRIPTION] {desc_clean}"
        elif subject_clean:
            return f"[SUBJECT] {subject_clean}"
        elif desc_clean:
            return f"[DESCRIPTION] {desc_clean}"
        else:
            return ""
    
    def extract_key_phrases(self, text: str) -> list:
        """
        Extract potential key phrases for feature augmentation.
        
        This is useful for hybrid TF-IDF + embedding approaches.
        
        Args:
            text: Preprocessed text
            
        Returns:
            List of potential key phrases
        """
        if not text:
            return []
        
        # Simple n-gram extraction (could be enhanced with RAKE/KeyBERT)
        words = text.split()
        phrases = []
        
        # Unigrams
        phrases.extend(words)
        
        # Bigrams
        for i in range(len(words) - 1):
            phrases.append(f"{words[i]} {words[i+1]}")
        
        return phrases
    
    def get_text_statistics(self, text: str) -> dict:
        """
        Compute text statistics for EDA and quality monitoring.
        
        Args:
            text: Raw or cleaned text
            
        Returns:
            Dictionary of text statistics
        """
        if not text:
            return {
                "char_count": 0,
                "word_count": 0,
                "avg_word_length": 0,
                "contains_url": False,
                "contains_email": False
            }
        
        words = text.split()
        
        return {
            "char_count": len(text),
            "word_count": len(words),
            "avg_word_length": sum(len(w) for w in words) / max(len(words), 1),
            "contains_url": bool(self._url_pattern.search(text)),
            "contains_email": bool(self._email_pattern.search(text))
        }
