
# CodeGuard — Software Requirements Specification

- **Document status:** Initial requirements draft.
- **Next review:** Architecture design.

---

## 1. Project Overview

CodeGuard is an AI-powered GitHub Pull Request Reviewer.

It analyzes pull request changes, identifies potentially risky code, retrieves relevant repository context, detects selected code-quality and security issues, and generates evidence-backed review recommendations.

The primary objective is to help developers understand the potential impact of code changes before merging a pull request.

## 2. Project Goals

- Retrieve and analyze GitHub pull requests.
- Understand changed files, functions, and classes.
- Perform deterministic static code analysis.
- Retrieve relevant repository context.
- Identify potential bugs and security concerns.
- Generate structured AI review findings.
- Recommend relevant tests.
- Produce comprehensive review reports.
- Evaluate the quality and reliability of AI reviews.
- Provide a usable dashboard for developers.

## 3. Target Users

- Software developers.
- Backend engineers.
- Code reviewers.
- Engineering teams.
- Technical leads.

## 4. Functional Requirements

### FR-01: GitHub Integration

- Connect GitHub repositories securely.
- Retrieve repository metadata.
- Retrieve pull request information.
- Retrieve changed filenames and code diffs.
- Support pagination and API rate limits.
- Retrieve relevant base and head file versions.

### FR-02: Diff Processing

- Parse unified Git diffs.
- Identify added, removed, and modified lines.
- Identify changed files.
- Preserve valid source-line and diff-position information.
- Handle deleted, renamed, binary, and unavailable files gracefully.

### FR-03: Static Code Analysis

Initially support Python repositories.

- Parse Python source using AST.
- Identify functions and classes.
- Extract imports.
- Identify recognizable local function calls.
- Detect selected risky code patterns.
- Identify changed functions and classes.
- Preserve file paths and source-line references.

### FR-04: Repository Context Retrieval

- Retrieve related function definitions.
- Identify callers where feasible.
- Inspect imported dependencies.
- Locate relevant test files.
- Retrieve related implementation patterns.
- Limit retrieved context to relevant evidence.

### FR-05: AI-Powered Code Review

Generate structured findings containing:

- Finding title.
- Severity.
- Confidence.
- File path.
- Line reference when verified.
- Problem description.
- Supporting code evidence.
- Why the issue matters.
- Suggested fix.
- Recommended validation.

All model outputs must be validated using Pydantic.

### FR-06: Finding Validation

- Verify referenced files exist in the analysis snapshot.
- Validate evidence against retrieved source code.
- Validate source-line references.
- Reject unsupported findings.
- Distinguish confirmed issues from potential AI suggestions.
- Avoid presenting uncertain suspicions as confirmed bugs.

### FR-07: Test Recommendations

- Identify relevant existing tests.
- Recommend unit tests.
- Recommend integration tests.
- Recommend edge-case tests.
- Recommend regression tests.
- Explain the code change motivating each recommendation.

### FR-08: Risk Assessment

- Calculate deterministic risk signals.
- Consider severity and affected components.
- Produce an overall PR risk assessment.
- Explain the reasons behind the assessment.

### FR-09: Review Reports

Generate reports containing:

- PR summary.
- Files analyzed.
- Affected functions and components.
- Findings grouped by severity.
- Supporting evidence.
- Recommended tests.
- Risk assessment.
- Analysis limitations.
- Completion status.

### FR-10: Database and History

- Persist PR analysis records.
- Persist review findings.
- Store processing status.
- Retrieve previous analysis reports.
- Track analysis metadata.

### FR-11: GitHub Webhooks

- Receive supported PR events.
- Verify webhook signatures.
- Prevent duplicate event processing.
- Trigger analysis through background processing.
- Track processing status and failures.

### FR-12: Review Publishing

- Generate GitHub-compatible review comments.
- Map findings to valid diff positions.
- Require explicit human approval.
- Prevent duplicate publishing.
- Never automatically approve or merge pull requests.

### FR-13: Evaluation

Measure:

- Finding precision and recall.
- False-positive rate.
- Severity accuracy.
- Evidence accuracy.
- Line-reference accuracy.
- Test recommendation quality.
- Analysis latency.
- Token consumption.
- Cost per reviewed PR.

## 5. Non-Functional Requirements

### Security

- Store secrets outside source control.
- Use least-privilege GitHub permissions.
- Verify GitHub webhooks.
- Validate all external input.
- Redact sensitive information.
- Protect against prompt injection.
- Treat repository content as untrusted data.
- Never execute arbitrary repository code.

### Reliability

- Implement API timeouts.
- Support bounded retries.
- Handle rate limits.
- Implement idempotent processing.
- Handle malformed AI responses.
- Support graceful failure recovery.
- Maintain structured logs.

### Performance

- Limit repository context size.
- Control LLM token consumption.
- Avoid unnecessary repeated API calls.
- Apply concurrency limits.
- Measure analysis latency.

### Maintainability

- Use modular architecture.
- Use Python type hints.
- Use Pydantic validation.
- Maintain clear separation of responsibilities.
- Use database migrations.
- Maintain automated tests and documentation.

### Testability

- Unit tests.
- Integration tests.
- API tests.
- Security tests.
- GitHub API mock tests.
- AI provider tests.
- Evaluation and regression tests.

## 6. Initial MVP Scope

The initial MVP will support:

1. Manual GitHub PR analysis.
2. Python repositories.
3. PR metadata and diff retrieval.
4. Changed-file classification.
5. Python AST analysis.
6. Repository context retrieval.
7. Mock AI provider for testing.
8. Structured review schemas.
9. Evidence validation.
10. Review report generation.

## 7. Out of Scope for the Initial MVP

- Automatic webhook processing.
- Multi-agent orchestration.
- Automatic GitHub review publishing.
- Automatic PR approval or merging.
- Multi-language analysis beyond the initial Python implementation.
- Vector database and embedding-based retrieval.
- Kubernetes deployment.
- Arbitrary repository code execution.

These may be introduced in later milestones when justified.

## 8. Technical Constraints

- Python 3.12.
- FastAPI backend.
- Next.js and TypeScript frontend.
- PostgreSQL database.
- GitHub REST API.
- Python AST.
- Pydantic.
- SQLAlchemy and Alembic.
- Pytest.
- Docker and Docker Compose.
- GitHub Actions.

Use additional technologies only when they solve a concrete engineering problem.

## 9. Engineering Principles

1. Deterministic checks must handle verifiable analysis.
2. LLMs must assist with reasoning, not invent repository facts.
3. Every finding must have supporting evidence.
4. AI-generated output must pass validation.
5. Repository code must never be executed automatically.
6. Human approval is required before publishing reviews.
7. Every major feature must have automated tests.
8. Production readiness must be verified, not assumed.

## 10. Definition of MVP Completion

The MVP is complete when:

- A real GitHub PR can be retrieved.
- Its changed files and diffs can be processed.
- Changed Python functions and classes can be identified.
- Relevant repository context can be retrieved.
- Static analysis produces verifiable results.
- The mock review provider produces valid structured output.
- Findings are checked against actual repository evidence.
- Test recommendations are grounded in code changes.
- A comprehensive review report is generated.
- Relevant automated tests pass.

---

**Document status:** Initial requirements draft.  
**Next review:** Architecture design.
