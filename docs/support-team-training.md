# AstroShield Support Team Training Guide

## Introduction

This document provides the AstroShield support team with procedures, tools, and knowledge to effectively troubleshoot common issues that customers may encounter. It covers key system components, common failure modes, diagnostic approaches, and resolution strategies.

## System Architecture Overview

AstroShield consists of the following primary components:

1. **Frontend UI** - Next.js-based web application
2. **Backend API** - FastAPI service providing core functionality
3. **Database** - PostgreSQL database storing spacecraft and conjunction data
4. **Redis Cache** - In-memory cache for frequently accessed data
5. **Kafka** - Messaging system for event-driven operations
6. **UDL Integration** - Interface with external UDL (Unified Data Library) service

Understanding these components and their interactions is essential for effective troubleshooting.

## Support Workflow

Follow this workflow for handling customer support requests:

1. **Triage**: Determine severity and impact
   - Critical: System unavailable or data integrity issues
   - High: Major feature unavailable
   - Medium: Feature degraded or poor performance
   - Low: Minor issues, UI glitches, or enhancement requests

2. **Gather Information**: Collect relevant data
   - Full error message and context
   - User actions leading to the issue
   - Timing of issue occurrence
   - Browser/device information (for UI issues)
   - User ID and account details
   - Affected NORAD IDs or spacecraft

3. **Investigate**: Use troubleshooting tools
   - Review application logs
   - Check monitoring dashboards
   - Examine database state if relevant
   - Verify system health status

4. **Resolve or Escalate**: Take appropriate action
   - Apply known fixes where applicable
   - Escalate to engineering when necessary
   - Document resolution for knowledge base

## Common Issues and Resolutions

### 1. Authentication and Access Issues

#### Symptom: User cannot log in

**Troubleshooting Steps**:
1. Verify user credentials exist in the database
2. Check for account lockout due to failed attempts
3. Ensure JWT secret is properly configured
4. Verify user has appropriate permissions

**Resolution Actions**:
- Reset user password if necessary
- Unlock account if locked
- Check JWT token expiration settings
- Update user permissions if incorrect

#### Symptom: "Unauthorized" errors when accessing API endpoints

**Troubleshooting Steps**:
1. Check if user token is expired
2. Verify token is being correctly sent in headers
3. Validate that user has required permissions for the operation
4. Check API logs for detailed authorization failures

**Resolution Actions**:
- Guide user to re-authenticate
- Update user permissions
- Check for clock synchronization issues if token validation fails

### 2. Historical Analysis Issues

#### Symptom: Historical analysis returns no data

**Troubleshooting Steps**:
1. Verify the NORAD ID exists in the system
2. Check date range validity
3. Examine database query logs for errors
4. Verify data exists for the requested period

**Resolution Actions**:
- Suggest alternative date ranges with known data
- Check if data import for that object failed
- Verify database connection and query performance

#### Symptom: Historical analysis times out or performs poorly

**Troubleshooting Steps**:
1. Check if date range is excessively large
2. Verify proper use of pagination parameters
3. Check server resource utilization
4. Look for slow queries in database logs

**Resolution Actions**:
- Advise user to use smaller date ranges
- Suggest appropriate pagination settings
- Restart services if resource utilization is high

### 3. Conjunction Data Issues

#### Symptom: Missing conjunction events

**Troubleshooting Steps**:
1. Verify UDL integration status
2. Check conjunction ingestion job logs
3. Confirm object is being properly tracked
4. Verify filtering parameters aren't excluding results

**Resolution Actions**:
- Check UDL connectivity
- Verify conjunction processing pipelines
- Update object catalog if necessary

#### Symptom: Incorrect conjunction data

**Troubleshooting Steps**:
1. Compare with source UDL data if available
2. Check for data processing errors in logs
3. Verify correct version of propagation models
4. Check for recent updates to object parameters

**Resolution Actions**:
- Reprocess data if corruption detected
- Update orbit models if outdated
- Reset cached data if inconsistency found

### 4. UI and Visualization Issues

#### Symptom: UI elements not loading or displaying correctly

**Troubleshooting Steps**:
1. Check browser console for JavaScript errors
2. Verify browser compatibility
3. Test with cache cleared
4. Check network requests for failed API calls

**Resolution Actions**:
- Guide user to clear browser cache
- Recommend supported browser version
- Check for CDN issues if static assets aren't loading

#### Symptom: 3D visualizations not rendering correctly

**Troubleshooting Steps**:
1. Verify WebGL support in user's browser
2. Check for graphics driver issues
3. Confirm data format is correct for visualization
4. Test with simpler visualization dataset

**Resolution Actions**:
- Recommend browser/hardware with better WebGL support
- Suggest disabling hardware acceleration if causing issues
- Provide alternative 2D visualization if available

### 5. Performance and Scaling Issues

#### Symptom: System slowdown during peak usage

**Troubleshooting Steps**:
1. Monitor system resource utilization
2. Check database connection pool status
3. Look for long-running queries or processes
4. Verify caching is functioning correctly

**Resolution Actions**:
- Restart overloaded services
- Scale up resources if needed
- Optimize problematic queries
- Clear and rebuild caches

#### Symptom: Timeout errors when processing large datasets

**Troubleshooting Steps**:
1. Check configured timeout values
2. Monitor memory usage during processing
3. Look for query optimization opportunities
4. Verify proper use of pagination

**Resolution Actions**:
- Advise on breaking requests into smaller chunks
- Adjust timeout settings if appropriate
- Schedule resource-intensive operations during off-peak hours

## Using Monitoring and Diagnostic Tools

### Health Check Endpoints

Use these endpoints to verify component status:

- `/health` - Overall service health
- `/health/db` - Database connection status
- `/health/cache` - Redis cache status
- `/health/udl` - UDL integration status

Example command:
```bash
curl -X GET "https://api.astroshield.com/health" -H "Authorization: Bearer SUPPORT_TOKEN"
```

### Log Access

Access logs through the centralized logging system:

1. Log into the Grafana dashboard at `https://monitor.astroshield.com`
2. Navigate to "Explore" -> "Logs"
3. Use the following query patterns:
   - `{service="backend"} |= "ERROR"` - Find backend errors
   - `{service="frontend"} |= "WARN"` - Find frontend warnings
   - `{service="api"} |= "user_id=USERXXXXX"` - Find logs for specific user

### Database Diagnostics

For database issues, use these queries:

1. Check for blocking transactions:
```sql
SELECT blocked_locks.pid AS blocked_pid,
       blocking_locks.pid AS blocking_pid,
       blocked_activity.usename AS blocked_user,
       blocking_activity.usename AS blocking_user,
       now() - blocked_activity.query_start AS blocked_duration
FROM pg_catalog.pg_locks blocked_locks
JOIN pg_catalog.pg_locks blocking_locks 
    ON blocking_locks.locktype = blocked_locks.locktype
    AND blocking_locks.DATABASE IS NOT DISTINCT FROM blocked_locks.DATABASE
    AND blocking_locks.relation IS NOT DISTINCT FROM blocked_locks.relation
    AND blocking_locks.page IS NOT DISTINCT FROM blocked_locks.page
    AND blocking_locks.tuple IS NOT DISTINCT FROM blocked_locks.tuple
    AND blocking_locks.virtualxid IS NOT DISTINCT FROM blocked_locks.virtualxid
    AND blocking_locks.transactionid IS NOT DISTINCT FROM blocked_locks.transactionid
    AND blocking_locks.classid IS NOT DISTINCT FROM blocked_locks.classid
    AND blocking_locks.objid IS NOT DISTINCT FROM blocked_locks.objid
    AND blocking_locks.objsubid IS NOT DISTINCT FROM blocked_locks.objsubid
    AND blocking_locks.pid != blocked_locks.pid
JOIN pg_catalog.pg_stat_activity blocked_activity ON blocked_activity.pid = blocked_locks.pid
JOIN pg_catalog.pg_stat_activity blocking_activity ON blocking_activity.pid = blocking_locks.pid
WHERE NOT blocked_locks.GRANTED;
```

2. Find slow queries:
```sql
SELECT pid, age(clock_timestamp(), query_start), usename, query
FROM pg_stat_activity
WHERE state != 'idle' AND query_start < now() - interval '5 minutes'
ORDER BY query_start;
```

### Cache Diagnostics

To check Redis cache status:

```bash
redis-cli -h redis.astroshield.com info stats
redis-cli -h redis.astroshield.com keys "conjunction:*"
```

## Escalation Procedures

Follow these guidelines for escalation:

1. **Tier 1 Support (First Response)**
   - Handle common issues documented in this guide
   - Gather information and perform initial troubleshooting
   - Resolve within 4 hours or escalate to Tier 2

2. **Tier 2 Support (Application Specialists)**
   - Address complex application and data issues
   - Deep dive into logs and monitoring data
   - Resolve within 8 hours or escalate to Tier 3

3. **Tier 3 Support (Engineering Team)**
   - Tackle system-level issues requiring code changes
   - Investigate data integrity or architectural problems
   - Provide temporary workarounds when necessary

### Critical Issue Protocol

For severity 1 (critical) issues:

1. Immediately notify the on-call engineer via PagerDuty
2. Create a dedicated Slack channel for the incident
3. Schedule a war room call if not resolved within 30 minutes
4. Provide hourly status updates to affected customers
5. Document all investigation steps in the incident report

## Support Team Tools and Resources

### Knowledge Base

Access the support knowledge base at: `https://support.astroshield.com/kb`

Important articles:
- KB001: Authentication Troubleshooting
- KB002: Common UDL Integration Issues
- KB003: Database Performance Optimization
- KB004: Historical Analysis Troubleshooting
- KB005: Interpreting System Metrics

### Support Ticket Guidelines

When creating tickets:
1. Use descriptive titles that summarize the issue
2. Include the customer organization and contact info
3. Specify the environment (production, staging)
4. Document all reproduction steps
5. Attach relevant logs and screenshots
6. Set appropriate severity and priority
7. Link related tickets if applicable

### Customer Communication Templates

Use standardized templates for common scenarios:

1. **Initial Response Template**:
```
Hello [Customer Name],

Thank you for contacting AstroShield Support. We've received your request regarding [brief issue description] and are investigating this issue (Ticket #[TICKET-ID]).

We'll update you with our findings or if we need additional information.

Best regards,
[Your Name]
AstroShield Support
```

2. **Status Update Template**:
```
Hello [Customer Name],

We're still investigating your issue with [brief description]. Our team has [current status/findings]. 

The next steps in our investigation are [next steps].

We expect to provide another update by [timeframe].

Best regards,
[Your Name]
AstroShield Support
```

3. **Resolution Template**:
```
Hello [Customer Name],

Good news! We've resolved the issue with [brief description]. The cause was [simple explanation of cause] and we've [explanation of solution].

To prevent this from happening again, we recommend [preventative advice if applicable].

Is there anything else we can help you with?

Best regards,
[Your Name]
AstroShield Support
```

## Training Certification

To complete training, support team members must:

1. Complete the self-paced training modules in the LMS
2. Pass the troubleshooting scenario quiz with at least 85%
3. Successfully handle 5 mock support tickets
4. Shadow an experienced support engineer for at least 3 real customer issues
5. Demonstrate ability to use all diagnostic tools

After certification, regular refresher training will be conducted quarterly to ensure knowledge of new features and resolution techniques stays current.

## Appendix: System Error Codes Reference

| Error Code | Description | Common Causes | Resolution |
|------------|-------------|--------------|------------|
| AUTH001 | Authentication Failed | Invalid credentials, account locked | Check credentials, unlock account |
| AUTH002 | Token Expired | Session timeout, clock skew | Re-authenticate, check system time |
| AUTH003 | Insufficient Permissions | User lacks required role | Adjust user permissions |
| DATA001 | Object Not Found | Invalid NORAD ID, data not imported | Verify object exists, check import status |
| DATA002 | Invalid Date Range | End date before start date, future dates | Correct date parameters |
| DATA003 | Pagination Error | Invalid page/size values | Adjust pagination parameters |
| PROC001 | Processing Timeout | Data too large, system overload | Reduce request scope, check system load |
| PROC002 | Calculation Error | Invalid inputs, algorithm failure | Verify inputs, check calculation logs |
| CONN001 | UDL Connection Failed | Network issue, auth failure | Check network, verify credentials |
| CONN002 | Database Connection Error | DB overload, connection limit | Check DB status, connection pool |
| SYS001 | Rate Limit Exceeded | Too many requests | Advise on rate limiting policies |
| SYS002 | System Maintenance | Scheduled downtime | Inform about maintenance schedule | 