# Custom Domain Configuration Guide

This document outlines the Domain Name System (DNS) configuration required to attach a custom domain to the AquaSense AI Vercel deployment.

## Production Domain Records

To configure a custom domain (e.g., `aquasense.example.org` or apex domain), add the following DNS records with your registrar or DNS provider:

### Apex Domain (e.g., `example.org`)
| Record Type | Host / Name | Target / Value | TTL |
| :--- | :--- | :--- | :--- |
| A | `@` | `76.76.21.21` | 3600 |

### Subdomain (e.g., `aquasense.example.org`)
| Record Type | Host / Name | Target / Value | TTL |
| :--- | :--- | :--- | :--- |
| CNAME | `aquasense` | `cname.vercel-dns.com.` | 3600 |

## Automated Domain Alias via Vercel CLI
Once DNS records are propagated and the Vercel CLI is authenticated:
```bash
# Add domain to project
vercel domains add <your-domain>

# Assign deployment alias
vercel alias set <deployment-url> <your-domain>
```

## Verification
Confirm the domain is live using HTTP head request:
```bash
curl -I https://<your-domain>/health
```
Do not claim the domain is live until `curl -I` returns an HTTP 200 OK status.
