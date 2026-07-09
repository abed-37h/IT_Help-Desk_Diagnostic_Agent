from app.tools.schema import Report, Error


def format_report(report: Report | Error) -> str:
    """
    Formats a structured Report into a human-readable string for display in the UI.
    If an Error is passed, returns a formatted error message instead.
    """

    if isinstance(report, Error):
        return (
            f"⚠️ Could not generate report.\n"
            f"Reason: {report.message}"
        )

    lines = []

    # ── Header ────────────────────────────────────────────────────────────────
    lines.append("=" * 60)
    lines.append("        IT HELP-DESK — SESSION REPORT")
    lines.append("=" * 60)

    # ── Ticket Info ───────────────────────────────────────────────────────────
    lines.append(f"\n📋 TICKET DETAILS")
    lines.append(f"  Ticket ID   : {report.ticket_id}")
    lines.append(f"  Status      : {report.ticket.status.replace('_', ' ').title()}")
    lines.append(f"  Opened      : {report.ticket.created_at}")
    if report.ticket.resolved_at:
        lines.append(f"  Resolved    : {report.ticket.resolved_at}")

    # ── User Info ─────────────────────────────────────────────────────────────
    lines.append(f"\n👤 USER")
    lines.append(f"  Name        : {report.user.user_name}")
    lines.append(f"  ID          : {report.user.user_id}")

    # ── Issue Info ────────────────────────────────────────────────────────────
    lines.append(f"\n🔍 ISSUE")
    lines.append(f"  Title       : {report.issue.title}")
    lines.append(f"  Category    : {report.issue.category}")
    lines.append(f"  Severity    : {report.issue.severity.title()}")

    # ── Troubleshooting Steps ─────────────────────────────────────────────────
    lines.append(f"\n🛠️  TROUBLESHOOTING STEPS")
    if report.steps_provided:
        for i, step in enumerate(report.steps_provided, start=1):
            lines.append(f"  {i}. {step}")
    else:
        lines.append("  No steps recorded.")

    # ── Resolution Notes ──────────────────────────────────────────────────────
    if report.ticket.resolution_notes:
        lines.append(f"\n📝 RESOLUTION NOTES")
        lines.append(f"  {report.ticket.resolution_notes}")

    # ── Handoff ───────────────────────────────────────────────────────────────
    if report.handoff_required:
        lines.append(f"\n🚨 ESCALATION REQUIRED")
        lines.append(f"  This issue requires follow-up from a human technician.")
        lines.append(f"  Please expect a technician to contact you shortly.")

    # ── Footer ────────────────────────────────────────────────────────────────
    lines.append(f"\n  Generated   : {report.generated_at}")
    lines.append("=" * 60)

    return "\n".join(lines)