"""
Escalator Agent — Notifies the human when something needs attention.

Tiers:
- Tier 1 (email): Informational — PR merged, daily summary
- Tier 2 (email + urgent flag): Needs attention — reviewer question we can't answer
- Tier 3 (webhook/WhatsApp): Urgent — circuit breaker, rejection streak, permissions issue

Uses Mailgun for email (already in xiangqi.com stack).
Uses a generic webhook for WhatsApp/Slack/Telegram (configure URL in env).
"""

import json
import logging
import os
from dataclasses import dataclass

import requests

from agents.base import BaseAgent, MODEL_FAST
from config.prompts import ESCALATION_SYSTEM_PROMPT

logger = logging.getLogger("agents.escalator")


@dataclass
class EscalationEvent:
    tier: int  # 1, 2, or 3
    title: str
    detail: str
    ticket_id: int | None = None
    pr_url: str | None = None
    suggested_action: str = ""


COMPOSE_NOTIFICATION_PROMPT = """\
Compose a notification for the human operator of an automated Django contribution system.

Event:
- Tier: {tier} ({tier_label})
- Title: {title}
- Detail: {detail}
- Ticket: #{ticket_id}
- PR: {pr_url}

Write a concise notification. Include:
1. What happened (1 sentence)
2. Why it needs attention (1 sentence)
3. Suggested action (1 sentence)
4. Links to ticket/PR

Respond with JSON:
{{
    "subject": "<email subject line>",
    "body": "<notification body, plain text, under 200 words>",
    "urgency": "low" | "medium" | "high"
}}
"""

TIER_LABELS = {1: "Informational", 2: "Needs Attention", 3: "Urgent"}


class EscalatorAgent(BaseAgent):
    """
    Sends human notifications when the system needs attention.

    Channels:
    - Email via Mailgun API (env: MAILGUN_API_KEY, MAILGUN_DOMAIN, ESCALATION_EMAIL)
    - Webhook for WhatsApp/Slack/Telegram (env: ESCALATION_WEBHOOK_URL)
    """

    def __init__(self, api_key: str | None = None):
        super().__init__(
            name="escalator",
            system_prompt=ESCALATION_SYSTEM_PROMPT,
            model=MODEL_FAST,
            api_key=api_key,
            use_skill=False,
        )
        self.mailgun_key = os.environ.get("MAILGUN_API_KEY", "")
        self.mailgun_domain = os.environ.get("MAILGUN_DOMAIN", "")
        self.recipient_email = os.environ.get("ESCALATION_EMAIL", "")
        self.webhook_url = os.environ.get("ESCALATION_WEBHOOK_URL", "")

    def escalate(self, event: EscalationEvent) -> dict:
        """
        Send a notification for the given escalation event.

        Tier 1 → email only
        Tier 2 → email with urgent flag
        Tier 3 → email + webhook (WhatsApp/Slack)
        """
        tier_label = TIER_LABELS.get(event.tier, "Unknown")
        logger.info("Escalating [Tier %d — %s]: %s", event.tier, tier_label, event.title)

        # AI composes the notification (1 cheap Sonnet call)
        prompt = COMPOSE_NOTIFICATION_PROMPT.format(
            tier=event.tier, tier_label=tier_label,
            title=event.title, detail=event.detail,
            ticket_id=event.ticket_id or "N/A",
            pr_url=event.pr_url or "N/A",
        )

        response = self.think(user_message=prompt, temperature=0.2, max_tokens=512)

        if response.succeeded:
            subject = response.parsed.get("subject", event.title)
            body = response.parsed.get("body", event.detail)
        else:
            # Fallback: plain text
            subject = f"[Django Agent — Tier {event.tier}] {event.title}"
            body = f"{event.detail}\n\nSuggested action: {event.suggested_action}"

        results = {"tier": event.tier, "title": event.title}

        # Send email for all tiers
        if self.mailgun_key and self.recipient_email:
            email_ok = self._send_email(subject, body)
            results["email_sent"] = email_ok
        else:
            logger.warning("Email not configured (set MAILGUN_API_KEY, MAILGUN_DOMAIN, ESCALATION_EMAIL)")
            results["email_sent"] = False

        # Send webhook for Tier 3
        if event.tier >= 3 and self.webhook_url:
            webhook_ok = self._send_webhook(event, subject, body)
            results["webhook_sent"] = webhook_ok
        elif event.tier >= 3:
            logger.warning("Webhook not configured for Tier 3 (set ESCALATION_WEBHOOK_URL)")
            results["webhook_sent"] = False

        return results

    def _send_email(self, subject: str, body: str) -> bool:
        """Send email via Mailgun API."""
        try:
            resp = requests.post(
                f"https://api.mailgun.net/v3/{self.mailgun_domain}/messages",
                auth=("api", self.mailgun_key),
                data={
                    "from": f"Django Agent <agent@{self.mailgun_domain}>",
                    "to": [self.recipient_email],
                    "subject": subject,
                    "text": body,
                },
                timeout=15,
            )
            resp.raise_for_status()
            logger.info("Email sent to %s", self.recipient_email)
            return True
        except requests.RequestException as e:
            logger.error("Email failed: %s", e)
            return False

    def _send_webhook(self, event: EscalationEvent, subject: str, body: str) -> bool:
        """Send webhook (WhatsApp/Slack/Telegram)."""
        payload = {
            "tier": event.tier,
            "subject": subject,
            "body": body,
            "ticket_id": event.ticket_id,
            "pr_url": event.pr_url,
        }
        try:
            resp = requests.post(self.webhook_url, json=payload, timeout=15)
            resp.raise_for_status()
            logger.info("Webhook sent to %s", self.webhook_url)
            return True
        except requests.RequestException as e:
            logger.error("Webhook failed: %s", e)
            return False
