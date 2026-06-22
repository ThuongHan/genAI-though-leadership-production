from __future__ import annotations

from abc import ABC, abstractmethod


class BaseExtractor(ABC):
    """
    Abstract base class for all extractor methods.
    Every extractor must implement extract().
    This enforces a common interface so main.py can swap methods with one line.
    """

    @abstractmethod
    def extract(self, text: str, source_label: str) -> list[dict]:
        """
        Extract belief statements from a single text string.

        Args:
            text:         Source text to extract beliefs from.
            source_label: Label for the source (e.g. 'blog', 'linkedin_posts').

        Returns:
            List of belief dicts. Each dict must contain:
            - "belief":          the belief statement (str)
            - "category":        mission | strategy | domain_knowledge | values | stance
            - "source_quote":    verbatim short phrase from the text (str)
            - "source_document": the source_label (str)
        """
        pass

    def extract_from_posts(self, posts: list[dict]) -> list[dict]:
        """
        Run extract() on each LinkedIn post title.
        Subclasses can override this for batched handling.
        """
        all_beliefs = []

        for post in posts:
            title = str(post.get("Post title", "")).strip()
            if not title:
                continue
            beliefs = self.extract(text=title, source_label="linkedin_posts")
            all_beliefs.extend(beliefs)

        return all_beliefs