"""
Git & GitHub Tools — Pure Python tooling for repo management and PR creation.

Tools layer. No AI. Just subprocess wrappers for git and gh CLI.
"""

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("tools.git")


@dataclass
class CommandResult:
    returncode: int
    stdout: str
    stderr: str

    @property
    def success(self) -> bool:
        return self.returncode == 0


def run_cmd(cmd: list[str], cwd: str | Path | None = None, timeout: int = 120) -> CommandResult:
    """Run a shell command and return structured result."""
    logger.debug("Running: %s", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            cwd=str(cwd) if cwd else None, timeout=timeout,
        )
        if result.returncode != 0:
            logger.warning("Command failed (exit %d): %s\nstderr: %s", result.returncode, " ".join(cmd), result.stderr[:500])
        return CommandResult(
            returncode=result.returncode,
            stdout=result.stdout.strip(),
            stderr=result.stderr.strip(),
        )
    except subprocess.TimeoutExpired:
        logger.error("Command timed out after %ds: %s", timeout, " ".join(cmd))
        return CommandResult(returncode=-1, stdout="", stderr=f"Timeout after {timeout}s")
    except FileNotFoundError:
        logger.error("Command not found: %s", cmd[0])
        return CommandResult(returncode=-1, stdout="", stderr=f"Command not found: {cmd[0]}")


class GitClient:
    """Git operations on the Django repo."""

    def __init__(self, repo_path: Path):
        self.repo_path = repo_path

    def _git(self, *args: str, timeout: int = 120) -> CommandResult:
        return run_cmd(["git", *args], cwd=self.repo_path, timeout=timeout)

    def ensure_clean(self) -> CommandResult:
        """Ensure working directory is clean."""
        return self._git("status", "--porcelain")

    def checkout_main_and_pull(self) -> CommandResult:
        """Switch to main and pull latest."""
        result = self._git("checkout", "main")
        if not result.success:
            return result
        return self._git("pull", "--rebase", "origin", "main")

    def create_branch(self, branch_name: str) -> CommandResult:
        """Create and checkout a new branch from main."""
        self.checkout_main_and_pull()
        return self._git("checkout", "-b", branch_name)

    def add_and_commit(self, message: str) -> CommandResult:
        """Stage all changes and commit."""
        add_result = self._git("add", "-A")
        if not add_result.success:
            return add_result
        return self._git("commit", "-m", message)

    def push_branch(self, branch_name: str, remote: str = "origin") -> CommandResult:
        """Push branch to remote."""
        return self._git("push", "-u", remote, branch_name)

    def get_diff(self, against: str = "main") -> CommandResult:
        """Get diff of current branch against target."""
        return self._git("diff", against, "--", ".", timeout=30)

    def get_diff_stat(self, against: str = "main") -> CommandResult:
        """Get diff stat (files changed summary)."""
        return self._git("diff", "--stat", against, "--", ".")

    def get_current_branch(self) -> str:
        result = self._git("branch", "--show-current")
        return result.stdout if result.success else ""

    def stash_and_reset(self) -> CommandResult:
        """Emergency reset — stash everything and go back to main."""
        self._git("stash")
        return self._git("checkout", "main")


class GitHubClient:
    """GitHub operations via gh CLI."""

    def __init__(self, repo: str = "django/django"):
        self.repo = repo

    def _gh(self, *args: str, timeout: int = 60) -> CommandResult:
        return run_cmd(["gh", *args], timeout=timeout)

    def create_pr(
        self,
        title: str,
        body: str,
        branch: str,
        base: str = "main",
    ) -> CommandResult:
        """Create a pull request."""
        return self._gh(
            "pr", "create",
            "--repo", self.repo,
            "--title", title,
            "--body", body,
            "--head", branch,
            "--base", base,
        )

    def get_pr_comments(self, pr_number: int) -> CommandResult:
        """Fetch all comments on a PR."""
        return self._gh(
            "api",
            f"repos/{self.repo}/issues/{pr_number}/comments",
            "--jq", ".[].body",
        )

    def get_pr_reviews(self, pr_number: int) -> CommandResult:
        """Fetch all review comments on a PR."""
        return self._gh(
            "api",
            f"repos/{self.repo}/pulls/{pr_number}/reviews",
        )

    def add_pr_comment(self, pr_number: int, body: str) -> CommandResult:
        """Add a comment to a PR."""
        return self._gh(
            "pr", "comment", str(pr_number),
            "--repo", self.repo,
            "--body", body,
        )

    def list_open_prs(self, author: str | None = None) -> CommandResult:
        """List open PRs, optionally filtered by author."""
        args = ["pr", "list", "--repo", self.repo, "--json", "number,title,url,updatedAt"]
        if author:
            args.extend(["--author", author])
        return self._gh(*args)
