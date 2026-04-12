# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✅         |

## Reporting a Vulnerability

If you discover a security vulnerability in AlphaEvo, please report it
responsibly.

**Do NOT open a public issue.**

Instead, report privately via [GitHub Security Advisories](https://github.com/ZhuLinsen/alphaevo/security/advisories/new) with:

1. A description of the vulnerability
2. Steps to reproduce (if applicable)
3. Impact assessment

We will acknowledge receipt within **48 hours** and aim to provide a fix or
mitigation within **7 days** for critical issues.

## Scope

AlphaEvo is a **research tool** that processes publicly available market data.
It does not handle:

- Real trading or brokerage credentials
- Personal financial data
- User authentication tokens (beyond optional LLM API keys)

Security concerns most relevant to this project:

- **LLM prompt injection** — Malicious strategy descriptions could attempt to
  manipulate LLM behavior. The parser validates all LLM output against the
  Strategy DSL schema before execution.
- **Arbitrary code execution** — Alpha Factory sandboxes synthesized factor
  code. Only whitelisted operations are permitted.
- **Dependency supply chain** — We pin transitive dependencies and run
  `pip-audit` in CI (planned).
- **SQL injection** — All database access uses parameterized queries via
  Python's `sqlite3` module.

## Disclosure Policy

We follow [coordinated disclosure](https://en.wikipedia.org/wiki/Coordinated_vulnerability_disclosure).
Credit will be given to reporters unless anonymity is requested.
