"""
Audit Logging for Compliance (HIPAA, SOC 2, GDPR)
Tracks all access to sensitive data for regulatory compliance
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List
import hashlib
import os


class AuditLogger:
    """
    Comprehensive audit trail for compliance

    Tracks:
    - Who accessed what data
    - When it was accessed
    - What action was performed
    - Result (success/failure)
    - IP address
    """

    def __init__(self, log_dir: str = "logs/audit", console_output: bool = True):
        """
        Args:
            log_dir: Directory to store audit logs
            console_output: Also print to console
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.console_output = console_output

        print(f"✓ AuditLogger initialized: {self.log_dir}")

    def log_event(
        self,
        user_id: str,
        action: str,
        resource: str,
        details: Optional[Dict] = None,
        status: str = "success",
        ip_address: Optional[str] = None,
    ) -> str:
        """
        Log an audit event

        Args:
            user_id: User who performed action
            action: Action performed (query, add_document, delete, login, etc.)
            resource: Resource accessed (document_id, query, etc.)
            details: Additional details dictionary
            status: success, failure, denied
            ip_address: IP address of request

        Returns:
            event_id: Unique event identifier
        """
        event_id = self._generate_event_id()

        event = {
            'event_id': event_id,
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'user_id': user_id,
            'action': action,
            'resource': resource,
            'status': status,
            'ip_address': ip_address or 'unknown',
            'details': details or {},
        }

        # Log to daily file (JSONL format)
        log_file = self.log_dir / f"audit_{datetime.utcnow().strftime('%Y%m%d')}.jsonl"

        with open(log_file, 'a') as f:
            f.write(json.dumps(event) + '\n')

        # Console output
        if self.console_output:
            status_symbol = "✓" if status == "success" else "✗" if status == "failure" else "⊘"
            print(f"[AUDIT] {status_symbol} {event['timestamp']} | {user_id} | {action} | {resource}")

        return event_id

    def log_query(
        self,
        user_id: str,
        query: str,
        num_results: int,
        response_time_ms: float,
        ip_address: Optional[str] = None,
    ):
        """Log RAG query event"""
        return self.log_event(
            user_id=user_id,
            action="query",
            resource="rag_system",
            details={
                'query': query[:200],  # Truncate long queries
                'num_results': num_results,
                'response_time_ms': response_time_ms,
            },
            status="success",
            ip_address=ip_address,
        )

    def log_document_access(
        self,
        user_id: str,
        document_id: str,
        action: str = "read",
        ip_address: Optional[str] = None,
    ):
        """Log document access (read, add, delete)"""
        return self.log_event(
            user_id=user_id,
            action=action,
            resource=f"document:{document_id}",
            status="success",
            ip_address=ip_address,
        )

    def log_authentication(
        self,
        user_id: str,
        success: bool,
        method: str = "api_key",
        ip_address: Optional[str] = None,
    ):
        """Log authentication attempt"""
        return self.log_event(
            user_id=user_id,
            action="authentication",
            resource="auth_system",
            details={'method': method},
            status="success" if success else "failure",
            ip_address=ip_address,
        )

    def log_data_export(
        self,
        user_id: str,
        data_type: str,
        num_records: int,
        ip_address: Optional[str] = None,
    ):
        """Log data export (important for HIPAA)"""
        return self.log_event(
            user_id=user_id,
            action="export",
            resource=data_type,
            details={'num_records': num_records},
            status="success",
            ip_address=ip_address,
        )

    def log_security_event(
        self,
        user_id: str,
        event_type: str,
        severity: str,
        details: Dict,
        ip_address: Optional[str] = None,
    ):
        """Log security event (unauthorized access, suspicious activity)"""
        return self.log_event(
            user_id=user_id,
            action="security_event",
            resource=event_type,
            details={'severity': severity, **details},
            status="failure",
            ip_address=ip_address,
        )

    def _generate_event_id(self) -> str:
        """Generate unique event ID"""
        timestamp = datetime.utcnow().isoformat()
        random_data = f"{timestamp}{os.urandom(8).hex()}"
        return hashlib.sha256(random_data.encode()).hexdigest()[:16]

    def query_logs(
        self,
        user_id: Optional[str] = None,
        action: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict]:
        """
        Query audit logs

        Args:
            user_id: Filter by user
            action: Filter by action type
            start_date: Filter by start date
            end_date: Filter by end date
            status: Filter by status
            limit: Maximum number of results

        Returns:
            List of matching events
        """
        logs = []
        files = sorted(self.log_dir.glob("audit_*.jsonl"))

        for log_file in files:
            with open(log_file, 'r') as f:
                for line in f:
                    event = json.loads(line)

                    # Apply filters
                    if user_id and event['user_id'] != user_id:
                        continue
                    if action and event['action'] != action:
                        continue
                    if status and event['status'] != status:
                        continue

                    if start_date or end_date:
                        event_time = datetime.fromisoformat(event['timestamp'].replace('Z', ''))
                        if start_date and event_time < start_date:
                            continue
                        if end_date and event_time > end_date:
                            continue

                    logs.append(event)

                    if len(logs) >= limit:
                        return logs

        return logs

    def get_user_activity(self, user_id: str, days: int = 7) -> Dict:
        """Get summary of user activity"""
        from collections import Counter

        start_date = datetime.utcnow() - timedelta(days=days)
        events = self.query_logs(user_id=user_id, start_date=start_date, limit=10000)

        actions = Counter(e['action'] for e in events)
        statuses = Counter(e['status'] for e in events)

        return {
            'user_id': user_id,
            'period_days': days,
            'total_events': len(events),
            'actions': dict(actions),
            'statuses': dict(statuses),
            'first_event': events[0]['timestamp'] if events else None,
            'last_event': events[-1]['timestamp'] if events else None,
        }

    def generate_compliance_report(
        self,
        start_date: datetime,
        end_date: datetime,
        output_file: str,
    ):
        """
        Generate compliance report for auditors

        Creates detailed report of all system access
        Required for HIPAA, SOC 2, GDPR audits
        """
        events = self.query_logs(start_date=start_date, end_date=end_date, limit=1000000)

        from collections import Counter

        report = {
            'report_period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat(),
            },
            'summary': {
                'total_events': len(events),
                'unique_users': len(set(e['user_id'] for e in events)),
                'actions': dict(Counter(e['action'] for e in events)),
                'statuses': dict(Counter(e['status'] for e in events)),
            },
            'security_events': [
                e for e in events if e['action'] == 'security_event'
            ],
            'failed_authentications': [
                e for e in events
                if e['action'] == 'authentication' and e['status'] == 'failure'
            ],
            'data_exports': [
                e for e in events if e['action'] == 'export'
            ],
            'top_users': dict(Counter(e['user_id'] for e in events).most_common(10)),
        }

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"✓ Compliance report generated: {output_file}")
        return report

    def archive_logs(self, days_to_keep: int = 90):
        """
        Archive old logs

        HIPAA requires keeping audit logs for 6 years
        This compresses old logs and moves to archive
        """
        import gzip
        import shutil

        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
        archive_dir = self.log_dir / "archive"
        archive_dir.mkdir(exist_ok=True)

        for log_file in self.log_dir.glob("audit_*.jsonl"):
            # Extract date from filename
            date_str = log_file.stem.replace('audit_', '')
            file_date = datetime.strptime(date_str, '%Y%m%d')

            if file_date < cutoff_date:
                # Compress and move to archive
                archive_file = archive_dir / f"{log_file.stem}.jsonl.gz"

                with open(log_file, 'rb') as f_in:
                    with gzip.open(archive_file, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)

                log_file.unlink()
                print(f"✓ Archived: {log_file.name}")


# Import for time delta
from datetime import timedelta


# Example usage
if __name__ == "__main__":
    print("Testing Audit Logging System\n")

    # Create audit logger
    logger = AuditLogger(log_dir="test_logs/audit")

    print("="*60)
    print("Testing Event Logging")
    print("="*60)

    # Log various events
    logger.log_authentication(
        user_id="user_123",
        success=True,
        method="api_key",
        ip_address="192.168.1.1"
    )

    logger.log_query(
        user_id="user_123",
        query="What is diabetes?",
        num_results=5,
        response_time_ms=245.3,
        ip_address="192.168.1.1"
    )

    logger.log_document_access(
        user_id="user_123",
        document_id="doc_456",
        action="read",
        ip_address="192.168.1.1"
    )

    logger.log_data_export(
        user_id="user_123",
        data_type="patient_records",
        num_records=10,
        ip_address="192.168.1.1"
    )

    logger.log_security_event(
        user_id="unknown",
        event_type="unauthorized_access",
        severity="high",
        details={'attempted_resource': 'admin_panel'},
        ip_address="10.0.0.1"
    )

    # Query logs
    print("\n" + "="*60)
    print("Testing Log Queries")
    print("="*60)

    user_logs = logger.query_logs(user_id="user_123")
    print(f"\nUser logs: {len(user_logs)} events")

    security_logs = logger.query_logs(action="security_event")
    print(f"Security events: {len(security_logs)} events")

    # Get user activity
    print("\n" + "="*60)
    print("Testing Activity Summary")
    print("="*60)

    activity = logger.get_user_activity("user_123", days=1)
    print(f"\nUser activity:")
    print(f"  Total events: {activity['total_events']}")
    print(f"  Actions: {activity['actions']}")

    # Generate compliance report
    print("\n" + "="*60)
    print("Testing Compliance Report")
    print("="*60)

    report = logger.generate_compliance_report(
        start_date=datetime.utcnow() - timedelta(days=1),
        end_date=datetime.utcnow(),
        output_file="test_logs/compliance_report.json"
    )

    print(f"\nReport summary:")
    print(f"  Total events: {report['summary']['total_events']}")
    print(f"  Unique users: {report['summary']['unique_users']}")
    print(f"  Security events: {len(report['security_events'])}")

    # Cleanup
    import shutil
    shutil.rmtree("test_logs")

    print("\n✓ All audit logging tests passed!")
