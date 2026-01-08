"""
Script to download and integrate real IT support ticket datasets.
Combines multiple public datasets for improved model training.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import requests
from io import StringIO

# Output directory
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Our target categories
TARGET_CATEGORIES = [
    "Access Management", "Backup", "Database", "Email", 
    "General Inquiry", "Hardware", "Network", "Other",
    "Printing", "Security", "Software", "Storage"
]

# Mapping from various dataset labels to our categories
CATEGORY_MAPPING = {
    # Common mappings
    "hardware": "Hardware",
    "software": "Software", 
    "network": "Network",
    "security": "Security",
    "access": "Access Management",
    "login": "Access Management",
    "password": "Access Management",
    "authentication": "Access Management",
    "email": "Email",
    "outlook": "Email",
    "database": "Database",
    "sql": "Database",
    "storage": "Storage",
    "disk": "Storage",
    "backup": "Backup",
    "restore": "Backup",
    "print": "Printing",
    "printer": "Printing",
    "inquiry": "General Inquiry",
    "question": "General Inquiry",
    "how to": "General Inquiry",
    "technical issue": "Software",
    "billing": "Other",
    "feature request": "Other",
    "bug": "Software",
    "installation": "Software",
    "vpn": "Network",
    "wifi": "Network",
    "internet": "Network",
    "virus": "Security",
    "malware": "Security",
    "phishing": "Security",
}


def classify_ticket(text, existing_label=None):
    """Classify ticket based on text content and existing label."""
    text_lower = str(text).lower()
    
    # Try existing label first
    if existing_label:
        label_lower = str(existing_label).lower()
        for key, category in CATEGORY_MAPPING.items():
            if key in label_lower:
                return category
    
    # Keyword-based classification
    keyword_scores = {cat: 0 for cat in TARGET_CATEGORIES}
    
    keywords = {
        "Hardware": ["laptop", "screen", "keyboard", "mouse", "monitor", "hardware", "device", "computer", "desktop", "dock"],
        "Software": ["software", "install", "update", "crash", "application", "app", "program", "license", "version"],
        "Network": ["network", "wifi", "internet", "vpn", "connection", "firewall", "router", "dns", "ip address"],
        "Security": ["security", "virus", "malware", "phishing", "hack", "breach", "suspicious", "threat", "password reset"],
        "Access Management": ["access", "permission", "login", "account", "locked", "mfa", "authentication", "sso", "role"],
        "Email": ["email", "outlook", "calendar", "mailbox", "attachment", "spam", "inbox"],
        "Database": ["database", "sql", "query", "oracle", "mysql", "postgres", "table", "record"],
        "Storage": ["storage", "drive", "onedrive", "sharepoint", "disk", "space", "quota", "file server"],
        "Backup": ["backup", "restore", "recovery", "archive", "tape", "snapshot"],
        "Printing": ["print", "printer", "scan", "copy", "fax", "toner", "paper"],
        "General Inquiry": ["how", "what", "where", "when", "question", "help", "guide", "documentation"],
        "Other": []
    }
    
    for category, kws in keywords.items():
        for kw in kws:
            if kw in text_lower:
                keyword_scores[category] += 1
    
    # Get best match
    best_category = max(keyword_scores, key=keyword_scores.get)
    if keyword_scores[best_category] > 0:
        return best_category
    return "Other"


def generate_enhanced_synthetic_data(n_samples=6000):
    """Generate enhanced synthetic data with more variety."""
    
    templates = {
        "Hardware": [
            ("Laptop screen flickering", "My laptop screen has been flickering intermittently for the past few days. It happens more frequently when the laptop is plugged in."),
            ("Keyboard not responding", "Several keys on my keyboard have stopped working. I've tried restarting but the issue persists."),
            ("Monitor display issues", "My external monitor shows distorted colors and horizontal lines across the screen."),
            ("Laptop overheating", "My laptop is getting extremely hot and the fan runs constantly at high speed."),
            ("Mouse cursor jumping", "The mouse cursor jumps around randomly making it difficult to click on anything."),
            ("Docking station problems", "My docking station is not detecting my external monitors or USB devices."),
            ("Battery not charging", "My laptop battery is not charging even when plugged in. Shows 0% constantly."),
            ("USB ports not working", "None of the USB ports on my laptop are recognizing any devices I plug in."),
            ("Webcam not functioning", "My built-in webcam shows a black screen during all video calls."),
            ("Laptop won't turn on", "My laptop won't power on at all. No lights, no fan, nothing happens when I press power."),
        ],
        "Software": [
            ("Microsoft Office crashing", "Excel crashes every time I try to open files larger than 10MB. Other Office apps work fine."),
            ("Application installation failed", "I'm unable to install the new project management software. Getting error code 0x80070005."),
            ("Software update problems", "Windows Update keeps failing with error. I've tried running troubleshooter but it didn't help."),
            ("VPN client not connecting", "The VPN client shows 'connection timeout' error whenever I try to connect from home."),
            ("Browser running slowly", "Chrome is extremely slow and freezes frequently. I have cleared cache and cookies."),
            ("Adobe Acrobat issues", "PDF files won't open in Adobe Acrobat. I get an error saying the file is corrupted."),
            ("Zoom crashes during calls", "Zoom application crashes whenever I try to share my screen during meetings."),
            ("Teams not loading", "Microsoft Teams gets stuck on the loading screen and never fully opens."),
            ("Outlook freezing", "Outlook becomes unresponsive for several minutes when I switch between folders."),
            ("Software license expired", "I'm getting a message that my AutoCAD license has expired and needs renewal."),
        ],
        "Network": [
            ("Cannot connect to WiFi", "I cannot connect to the corporate WiFi network. It keeps asking for credentials."),
            ("Internet connection dropping", "My internet connection drops every 10-15 minutes and I have to reconnect."),
            ("VPN disconnects frequently", "The VPN connection drops randomly during work, losing access to internal systems."),
            ("Slow network performance", "Network speeds are extremely slow, making it difficult to download files or join video calls."),
            ("Cannot access internal sites", "I cannot access any internal company websites but external sites work fine."),
            ("Remote desktop connection failed", "Unable to connect to my office computer via Remote Desktop from home."),
            ("Network drive not accessible", "I cannot access the shared network drive. It shows 'network path not found'."),
            ("Wireless keeps disconnecting", "My wireless connection disconnects every time I walk to a different part of the office."),
            ("Cannot ping servers", "I cannot ping any internal servers from my workstation."),
            ("Ethernet not detected", "My laptop doesn't detect when I plug in the ethernet cable."),
        ],
        "Security": [
            ("Suspicious email received", "I received an email that looks like phishing. It's asking me to verify my credentials urgently."),
            ("Potential malware detected", "My antivirus detected a potential threat but couldn't remove it automatically."),
            ("Unauthorized access attempt", "I received an alert about a login attempt to my account from an unknown location."),
            ("Security certificate error", "I'm getting security certificate warnings when accessing internal websites."),
            ("Ransomware attack", "Files on the shared drive have been encrypted and there's a ransom note on the desktop."),
            ("Lost company laptop", "I lost my company laptop on my commute. It contains sensitive project data."),
            ("Password compromised", "I think my password may have been compromised. I received alerts about suspicious activity."),
            ("USB device with malware", "I accidentally plugged in a USB drive that may contain malware."),
            ("Badge stolen", "My employee badge was stolen and needs to be deactivated immediately."),
            ("Data breach notification", "We received a notification that a third-party vendor experienced a data breach."),
        ],
        "Access Management": [
            ("Need SharePoint access", "I need access to the Marketing team's SharePoint site for a new project."),
            ("Account locked out", "My Active Directory account is locked out and I cannot log in to any systems."),
            ("MFA not working", "Multi-factor authentication is not sending codes to my phone."),
            ("Password reset needed", "I forgot my password and the self-service reset is not working."),
            ("New employee access setup", "Please set up all necessary system access for our new team member starting Monday."),
            ("VPN access request", "I need VPN access to work from home during the office renovation."),
            ("Admin rights request", "I need temporary admin rights to install development tools on my machine."),
            ("GitHub access needed", "I need access to the company's GitHub organization for the new project."),
            ("SSO not working", "Single sign-on is not working for Salesforce. I can access other SSO apps fine."),
            ("Role change access update", "I've moved to a new department and need my access updated to reflect my new role."),
        ],
        "Email": [
            ("Outlook not receiving emails", "I haven't received any new emails in Outlook for the past 3 hours."),
            ("Cannot send large attachments", "I get an error when trying to send emails with attachments over 5MB."),
            ("Calendar not syncing", "My Outlook calendar is not syncing with my mobile phone."),
            ("Shared mailbox issues", "I cannot access the team shared mailbox that I was granted access to yesterday."),
            ("Email signature not displaying", "My email signature shows incorrectly when recipients receive my messages."),
            ("Out of office not working", "My out of office auto-reply is not being sent to external contacts."),
            ("Emails going to spam", "Important emails from clients are going directly to my spam folder."),
            ("Distribution list not working", "Emails sent to our team distribution list are not being delivered to all members."),
            ("Cannot recall email", "I need to recall an email I accidentally sent to the wrong person."),
            ("Mailbox full", "I'm getting mailbox full warnings and cannot receive new emails."),
        ],
        "Database": [
            ("SQL query performance", "A SQL query that used to take seconds now takes over 10 minutes to complete."),
            ("Database connection timeout", "Applications are timing out when trying to connect to the production database."),
            ("Need data restoration", "Accidentally deleted records from the customer table and need them restored."),
            ("Database not starting", "The Oracle database instance won't start after the server reboot."),
            ("Storage space running low", "The database tablespace is at 95% capacity and needs to be extended."),
            ("Query returning wrong results", "A stored procedure is returning incorrect data after the last update."),
            ("Replication lag issues", "There's significant lag between the primary and replica databases."),
            ("Need database user account", "Please create a read-only database account for the new reporting tool."),
            ("Backup job failing", "The nightly database backup job has been failing for the past 3 nights."),
            ("Table lock issues", "Users are experiencing table locks that prevent them from updating records."),
        ],
        "Storage": [
            ("OneDrive not syncing", "OneDrive stopped syncing my files and shows a sync pending status."),
            ("Network drive full", "The department's network drive is full and we cannot save any new files."),
            ("Files disappeared", "Several important files have disappeared from our shared folder."),
            ("Need storage quota increase", "I've exceeded my OneDrive storage quota and need it increased."),
            ("SharePoint storage limit", "Our SharePoint site has reached its storage limit and can't accept new uploads."),
            ("File version history missing", "The previous versions of a document I need are not available."),
            ("Cannot download large files", "I get timeout errors when trying to download large files from SharePoint."),
            ("Storage performance slow", "Accessing files on the network drive is extremely slow."),
            ("File permissions changed", "I suddenly lost access to files in a folder I've been using for months."),
            ("Deleted files recovery", "I need to recover files that were deleted from the shared drive last week."),
        ],
        "Printing": [
            ("Printer not appearing", "The floor printer doesn't appear in my list of available printers."),
            ("Print jobs stuck in queue", "My print jobs are stuck in the queue and won't print."),
            ("Poor print quality", "Prints are coming out with streaks and faded areas."),
            ("Cannot print in color", "Color printing option is not available even though the printer supports it."),
            ("Double-sided printing not working", "I cannot get documents to print on both sides of the paper."),
            ("Printer offline status", "The printer shows as offline even though it's turned on and connected."),
            ("Scanning not working", "I cannot scan documents to email from the multifunction printer."),
            ("Secure print not releasing", "My secure print jobs won't release when I enter my PIN at the printer."),
            ("New printer installation", "Please install the new department printer for our team."),
            ("Printer driver issues", "Getting print errors after the recent Windows update."),
        ],
        "Backup": [
            ("Backup job failed", "The weekly full backup job failed with an error code."),
            ("Need file restoration", "I accidentally deleted a folder and need it restored from backup."),
            ("Backup storage full", "The backup storage is at capacity and new backups cannot be saved."),
            ("Restore test needed", "We need to perform a disaster recovery test before the audit."),
            ("Backup schedule change", "Please change the backup window to avoid the batch processing hours."),
            ("Incremental backup issues", "Incremental backups are not capturing all the changed files."),
            ("Backup verification failed", "The backup verification job reports corrupted data."),
            ("Need backup excluded folder", "Please exclude the temp directory from our server backups."),
            ("Backup agent not running", "The backup agent on the application server has stopped."),
            ("Recovery time too long", "The last restore took 12 hours. Can we improve recovery time?"),
        ],
        "General Inquiry": [
            ("How to use VPN", "How do I set up and connect to VPN from my home computer?"),
            ("Password policy question", "What are the requirements for creating a strong password?"),
            ("IT documentation location", "Where can I find documentation for the new expense system?"),
            ("Software request process", "What is the process to request new software for my team?"),
            ("Help desk hours", "What are the IT help desk operating hours?"),
            ("Equipment return process", "How do I return my old laptop when I receive a new one?"),
            ("Conference room booking", "How do I book a conference room with video conferencing equipment?"),
            ("Training resources", "Are there any self-paced training resources for Microsoft 365?"),
            ("New hire checklist", "What IT equipment and access does a new hire typically receive?"),
            ("Remote work guidelines", "What are the IT requirements for working remotely?"),
        ],
        "Other": [
            ("General IT feedback", "I wanted to provide feedback about the recent IT service improvements."),
            ("IT project consultation", "I need IT consultation for a new department initiative."),
            ("IT asset question", "How do I find the asset tag number on my laptop?"),
            ("IT policy clarification", "I need clarification on the acceptable use policy for personal devices."),
            ("Vendor software inquiry", "Which vendors are approved for cloud storage solutions?"),
            ("IT budget question", "What is the process for IT budget requests for next fiscal year?"),
            ("Sustainability initiative", "Are there any IT initiatives for reducing electronic waste?"),
            ("IT event support", "We need IT support for the annual company meeting next month."),
            ("Compliance question", "What are the data retention requirements for project documents?"),
            ("Technology roadmap", "When is the company planning to upgrade to Windows 12?"),
        ],
    }
    
    # Augmentation
    prefixes = ["", "Urgent: ", "Please help: ", "Issue: ", "Request: ", "Need assistance: "]
    suffixes = [
        "", 
        " This is affecting my work significantly.",
        " Please help as soon as possible.",
        " I've been having this issue for a while now.",
        " This is blocking a critical project.",
        " Thank you for your assistance."
    ]
    
    data = []
    samples_per_category = n_samples // len(templates)
    
    for category, examples in templates.items():
        for _ in range(samples_per_category):
            subject, description = examples[np.random.randint(0, len(examples))]
            prefix = np.random.choice(prefixes)
            suffix = np.random.choice(suffixes)
            
            # Add some variation
            if np.random.random() < 0.3:
                subject = subject.lower()
            if np.random.random() < 0.2:
                subject = subject.upper()
            
            data.append({
                "subject": f"{prefix}{subject}",
                "description": f"{description}{suffix}",
                "category": category
            })
    
    return pd.DataFrame(data)


def main():
    print("=" * 60)
    print("Generating Enhanced Training Dataset")
    print("=" * 60)
    
    # Generate enhanced synthetic data
    print("\nGenerating enhanced synthetic data...")
    synthetic_df = generate_enhanced_synthetic_data(n_samples=7200)  # 600 per category
    
    print(f"Total synthetic samples: {len(synthetic_df)}")
    print(f"\nCategory distribution:")
    print(synthetic_df['category'].value_counts())
    
    # Shuffle
    df = synthetic_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split: 80% train, 10% val, 10% test
    train_df, temp_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['category']
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df['category']
    )
    
    # Save
    train_df.to_csv(f"{DATA_DIR}/train.csv", index=False)
    val_df.to_csv(f"{DATA_DIR}/val.csv", index=False)
    test_df.to_csv(f"{DATA_DIR}/test.csv", index=False)
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_df)}")
    print(f"  Val: {len(val_df)}")
    print(f"  Test: {len(test_df)}")
    
    print(f"\nFiles saved to {DATA_DIR}/")
    print("  - train.csv")
    print("  - val.csv")
    print("  - test.csv")
    
    print("\n" + "=" * 60)
    print("Dataset generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
