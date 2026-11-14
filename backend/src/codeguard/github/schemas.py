from datetime import datetime
from typing import Literal

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field


class GitHubBaseModel(BaseModel):
    """Base model for GitHub API responses."""

    model_config = ConfigDict(
        extra="ignore",
        populate_by_name=True,
    )


class GitHubUser(GitHubBaseModel):
    """GitHub user information."""

    login: str
    id: int | None = None
    avatar_url: AnyHttpUrl | None = None
    html_url: AnyHttpUrl | None = None
    type: str | None = None


class GitHubRepository(GitHubBaseModel):
    """GitHub repository metadata."""

    id: int
    name: str
    full_name: str
    private: bool
    html_url: AnyHttpUrl
    description: str | None = None
    default_branch: str | None = None
    owner: GitHubUser
    language: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class GitHubBranchReference(GitHubBaseModel):
    """Base or head branch information in a pull request."""

    ref: str
    sha: str
    label: str | None = None
    user: GitHubUser | None = None


class GitHubPullRequest(GitHubBaseModel):
    """GitHub pull request metadata."""

    url: AnyHttpUrl
    html_url: AnyHttpUrl
    number: int
    state: Literal["open", "closed"]
    title: str
    body: str | None = None
    draft: bool = False
    merged: bool | None = None
    mergeable: bool | None = None
    user: GitHubUser | None = None
    base: GitHubBranchReference
    head: GitHubBranchReference
    created_at: datetime | None = None
    updated_at: datetime | None = None


class GitHubPullRequestFile(GitHubBaseModel):
    """A file changed in a GitHub pull request."""

    sha: str | None = None
    filename: str
    status: Literal[
        "added",
        "removed",
        "modified",
        "renamed",
        "copied",
        "changed",
        "unchanged",
    ]
    additions: int = Field(ge=0)
    deletions: int = Field(ge=0)
    changes: int = Field(ge=0)

    blob_url: AnyHttpUrl | None = None
    raw_url: AnyHttpUrl | None = None
    contents_url: AnyHttpUrl | None = None

    patch: str | None = None
    previous_filename: str | None = None
