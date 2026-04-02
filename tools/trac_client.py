"""
Trac API Client — Pure Python tooling for Django's issue tracker.

This is the TOOLS layer. No AI here — just clean API access
that agents use to interact with Trac.
"""

import csv
import io
import logging
import re
import time
from dataclasses import dataclass, field

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger("tools.trac")

TRAC_BASE_URL = "https://code.djangoproject.com"


@dataclass
class TracComment:
    author: str
    text: str
    timestamp: str


@dataclass
class TracTicket:
    ticket_id: int
    summary: str
    component: str
    ticket_type: str
    severity: str
    version: str
    owner: str
    reporter: str
    status: str
    stage: str
    has_patch: bool
    needs_better_patch: bool
    needs_tests: bool
    needs_docs: bool
    easy_picking: bool
    description: str = ""
    created: str = ""
    modified: str = ""
    url: str = ""
    comments: list[TracComment] = field(default_factory=list)

    @property
    def trac_url(self) -> str:
        return self.url or f"{TRAC_BASE_URL}/ticket/{self.ticket_id}"

    def to_context_string(self) -> str:
        """Format ticket as a rich text block for an AI agent to read."""
        comments_text = ""
        if self.comments:
            comments_text = "\n\n--- COMMENT THREAD ---\n"
            for i, c in enumerate(self.comments, 1):
                comments_text += f"\nComment #{i} by {c.author} ({c.timestamp}):\n{c.text}\n"

        return f"""
TICKET #{self.ticket_id}: {self.summary}
URL: {self.trac_url}
Component: {self.component}
Type: {self.ticket_type}
Severity: {self.severity}
Status: {self.status}
Stage: {self.stage}
Owner: {self.owner or '(unassigned)'}
Reporter: {self.reporter}
Has patch: {self.has_patch}
Needs better patch: {self.needs_better_patch}
Needs tests: {self.needs_tests}
Needs docs: {self.needs_docs}
Easy picking: {self.easy_picking}
Created: {self.created}
Last modified: {self.modified}

--- DESCRIPTION ---
{self.description}
{comments_text}
""".strip()


class TracClient:
    """Client for Django's Trac issue tracker."""

    def __init__(self, rate_limit_seconds: float = 1.0):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "DjangoContributionAgent/1.0",
        })
        self.rate_limit = rate_limit_seconds

    def _throttle(self):
        time.sleep(self.rate_limit)

    def fetch_easy_pickings(self, include_needs_better_patch: bool = True) -> list[TracTicket]:
        """Fetch all open easy-picking tickets from Trac.

        Returns lightweight ticket objects (no description/comments yet).
        Call fetch_ticket_detail() to hydrate individual tickets.
        """
        all_tickets = []

        # Category 1: Easy pickings with no existing patch
        params = {
            "status": ["new", "assigned"],
            "easy": "1",
            "has_patch": "0",
            "format": "csv",
            "col": [
                "id", "summary", "component", "type", "severity",
                "version", "owner", "reporter", "status", "stage",
                "has_patch", "needs_better_patch", "needs_tests",
                "needs_docs", "modified", "created",
            ],
            "order": "priority",
        }
        all_tickets.extend(self._fetch_ticket_list(params))

        # Category 2: Easy pickings that need a better patch
        if include_needs_better_patch:
            params2 = {**params, "has_patch": "1", "needs_better_patch": "1"}
            all_tickets.extend(self._fetch_ticket_list(params2))

        # Deduplicate
        seen = set()
        unique = []
        for t in all_tickets:
            if t.ticket_id not in seen:
                seen.add(t.ticket_id)
                unique.append(t)

        logger.info("Fetched %d unique easy-picking tickets from Trac", len(unique))
        return unique

    def fetch_ticket_detail(self, ticket: TracTicket) -> TracTicket:
        """Hydrate a ticket with its full description and comment thread."""
        self._throttle()
        url = f"{TRAC_BASE_URL}/ticket/{ticket.ticket_id}"
        logger.debug("Fetching detail for #%d", ticket.ticket_id)

        resp = self.session.get(url, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Extract description
        desc_div = soup.find("div", class_="searchable")
        ticket.description = desc_div.get_text(strip=True) if desc_div else ""

        # Extract comments
        ticket.comments = []
        for change in soup.find_all("div", class_="change"):
            author = ""
            timestamp = ""

            author_link = change.find("a", class_="author")
            if author_link:
                author = author_link.get_text(strip=True)

            time_tag = change.find("a", class_="timeline")
            if time_tag:
                timestamp = time_tag.get("title", "")

            comment_div = change.find("div", class_="comment")
            text = comment_div.get_text(strip=True) if comment_div else ""

            if text or author:
                ticket.comments.append(TracComment(
                    author=author, text=text, timestamp=timestamp,
                ))

        ticket.url = url
        return ticket

    def _fetch_ticket_list(self, params: dict) -> list[TracTicket]:
        """Fetch ticket list from Trac CSV export."""
        url = self._build_query_url(params)
        logger.info("Querying Trac: %s", url)

        resp = self.session.get(url, timeout=30)
        resp.raise_for_status()

        # Trac CSV responses have a UTF-8 BOM; use utf-8-sig to strip it.
        # Column names from Trac are Title-cased and differ from query param names:
        #   id, Summary, Component, Type, Severity, Version, Owner, Reporter,
        #   Status, Triage Stage, Has patch, Patch needs improvement,
        #   Needs tests, Needs documentation
        text = resp.content.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(text))
        tickets = []
        for row in reader:
            try:
                ticket_id = int(row["id"])
            except (KeyError, ValueError) as e:
                logger.warning("Skipping row with missing/invalid id: %s (keys: %s)", e, list(row.keys()))
                continue
            tickets.append(TracTicket(
                ticket_id=ticket_id,
                summary=row.get("Summary", ""),
                component=row.get("Component", ""),
                ticket_type=row.get("Type", ""),
                severity=row.get("Severity", ""),
                version=row.get("Version", ""),
                owner=row.get("Owner", ""),
                reporter=row.get("Reporter", ""),
                status=row.get("Status", ""),
                stage=row.get("Triage Stage", ""),
                has_patch=row.get("Has patch", "0") == "1",
                needs_better_patch=row.get("Patch needs improvement", "0") == "1",
                needs_tests=row.get("Needs tests", "0") == "1",
                needs_docs=row.get("Needs documentation", "0") == "1",
                easy_picking=True,
                created=row.get("created", row.get("Created", "")),
                modified=row.get("modified", row.get("Modified", "")),
            ))

        logger.info("Got %d tickets from query", len(tickets))
        return tickets

    @staticmethod
    def _build_query_url(params: dict) -> str:
        parts = []
        for key, value in params.items():
            if isinstance(value, list):
                for v in value:
                    parts.append(f"{key}={v}")
            else:
                parts.append(f"{key}={value}")
        return f"{TRAC_BASE_URL}/query?{'&'.join(parts)}"
