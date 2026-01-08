"""
Script to download and prepare IT service desk ticket dataset.
Downloads from Hugging Face and creates train/val/test splits.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

# Create synthetic IT service desk ticket data based on realistic patterns
# This simulates what you would get from the HuggingFace datasets

np.random.seed(42)

# Define ticket categories and templates
CATEGORIES = {
    "Hardware": [
        "My laptop screen is flickering intermittently",
        "Keyboard keys are sticking and not registering properly",
        "Mouse cursor jumps around the screen randomly",
        "Laptop battery drains within 2 hours of use",
        "Monitor displays horizontal lines across the screen",
        "Docking station not recognizing external monitors",
        "Laptop overheating and shutting down unexpectedly",
        "Printer paper jam keeps recurring",
        "Webcam shows black screen during video calls",
        "Hard drive making clicking noises",
        "USB ports not detecting any connected devices",
        "Laptop fan running constantly at high speed",
        "External monitor flickering when connected via HDMI",
        "Bluetooth headset not connecting to laptop",
        "Laptop touchpad not responding to clicks",
    ],
    "Software": [
        "Microsoft Office keeps crashing when opening Excel files",
        "Unable to install the latest software updates",
        "Application freezes every time I try to save a document",
        "VPN client showing connection timeout errors",
        "Browser constantly redirecting to unknown websites",
        "Software license expired and needs renewal",
        "Application not compatible with current Windows version",
        "Outlook not syncing emails from server",
        "Adobe Acrobat cannot open PDF files",
        "Antivirus blocking legitimate application",
        "Zoom application crashes during screen sharing",
        "Teams notifications not appearing on desktop",
        "Software installation stuck at 50 percent",
        "Application requires administrator privileges to run",
        "Java runtime environment needs to be updated",
    ],
    "Network": [
        "Cannot connect to corporate WiFi network",
        "Internet connection drops every few minutes",
        "VPN disconnects randomly during work",
        "Unable to access shared network drives",
        "Slow internet speed affecting video calls",
        "Cannot ping internal servers from my machine",
        "Network printer not discoverable on the network",
        "WiFi password not accepted after password reset",
        "Ethernet connection not detected by laptop",
        "Cannot access external websites but internal works",
        "Network timeout when accessing company portal",
        "Wireless connection unstable in conference room",
        "Unable to connect to remote desktop server",
        "DNS resolution failing for internal domains",
        "Firewall blocking access to required ports",
    ],
    "Access Management": [
        "Need access to SharePoint site for new project",
        "Account locked out after multiple password attempts",
        "Request access to customer database system",
        "Multi-factor authentication not sending verification codes",
        "Unable to reset password through self-service portal",
        "Need VPN access for remote work setup",
        "Request access to JIRA project management tool",
        "Active Directory group membership needs update",
        "Single sign-on not working for cloud applications",
        "Need elevated permissions on development server",
        "Request access to GitHub organization repository",
        "Azure AD account synchronization issues",
        "Need access to company Slack workspace",
        "Request admin rights for software installation",
        "Unable to access email after department transfer",
    ],
    "Email": [
        "Outlook not receiving new emails",
        "Cannot send emails with large attachments",
        "Email signature not appearing correctly",
        "Calendar invites not syncing to mobile device",
        "Shared mailbox access not working",
        "Out of office auto-reply not activating",
        "Emails going to spam folder incorrectly",
        "Cannot add email account to mobile phone",
        "Email search not returning expected results",
        "Distribution list emails not being delivered",
        "Unable to recall sent email message",
        "Email rules not filtering messages correctly",
        "Calendar showing wrong time zone",
        "Cannot open encrypted email attachments",
        "Mail quota exceeded warning appearing",
    ],
    "Security": [
        "Suspicious email received asking for credentials",
        "Computer infected with malware after clicking link",
        "Unauthorized purchase made on company card",
        "Lost laptop containing sensitive company data",
        "Suspicious login attempt from unknown location",
        "Ransomware encrypted all files on shared drive",
        "USB drive with confidential data gone missing",
        "Unknown software installed without permission",
        "Employee badge stolen need immediate deactivation",
        "Received call from someone claiming to be IT",
        "Website certificate warning when accessing portal",
        "Data breach notification needs investigation",
        "Unauthorized access to sensitive customer data",
        "Phishing email clicked before realizing it was fake",
        "Security vulnerability found in internal application",
    ],
    "Database": [
        "SQL query running extremely slow",
        "Database connection timeout errors occurring",
        "Unable to connect to production database",
        "Need restoration of accidentally deleted data",
        "Database queries returning incorrect results",
        "Running out of database storage space",
        "Oracle database instance not starting",
        "Need new database user account created",
        "Database replication lag causing issues",
        "Table locks preventing data updates",
        "Need database schema changes deployed",
        "Backup job failing every night",
        "Database migration causing application errors",
        "Query optimization needed for report",
        "Connection pool exhausted for application",
    ],
    "General Inquiry": [
        "How do I set up email on my mobile device",
        "What is the password policy for company accounts",
        "Where can I find IT documentation and guides",
        "How to request new hardware for team member",
        "What is the process for software purchase approval",
        "How do I connect to VPN from home",
        "Where can I submit IT budget requests",
        "What are the operating hours for help desk",
        "How to schedule conference room equipment",
        "What software is approved for use",
        "How do I transfer files to new laptop",
        "What is the process for returning old equipment",
        "How to set up voicemail on desk phone",
        "Where to find training videos for new systems",
        "How do I request a guest WiFi account",
    ],
    "Storage": [
        "OneDrive not syncing files properly",
        "Running out of storage on network drive",
        "Files disappeared from shared folder",
        "Cannot access cloud storage from mobile",
        "Need increased mailbox storage quota",
        "SharePoint site storage limit reached",
        "Backup restoration taking too long",
        "File version history not available",
        "Cannot upload large files to Teams",
        "Storage migration causing access issues",
        "Deleted files not in recycle bin",
        "Need archival solution for old projects",
        "File permissions changed unexpectedly",
        "Storage performance degraded significantly",
        "Cannot access files from previous backup",
    ],
    "Printing": [
        "Printer not appearing in available printers list",
        "Print jobs stuck in queue and not printing",
        "Color printing not working correctly",
        "Unable to print double-sided documents",
        "Printer driver needs to be reinstalled",
        "Cannot scan documents to email",
        "Print quality is poor with streaky lines",
        "Secure print jobs not releasing at printer",
        "Need printer installed for new employee",
        "Printer showing offline status constantly",
        "Cannot find network printer after move",
        "PDF files printing blank pages",
        "Label printer not working correctly",
        "Print preview different from actual output",
        "Toner low warning but just replaced cartridge",
    ],
    "Backup": [
        "Backup job failed with error code",
        "Need to restore files from last week",
        "Backup taking longer than maintenance window",
        "Cannot locate backup tapes for audit",
        "Incremental backup not capturing changes",
        "Backup storage running critically low",
        "Disaster recovery test needs scheduling",
        "Backup agent not running on server",
        "Need to exclude folder from backup",
        "Backup report showing incomplete status",
        "Retention policy needs adjustment",
        "Cloud backup synchronization issues",
        "Need backup solution for new application",
        "Backup verification job failing",
        "Recovery point objective not being met",
    ],
    "Other": [
        "General IT inquiry not covered by categories",
        "Request for IT consultation on project",
        "Feedback on IT services quality",
        "Suggestion for IT process improvement",
        "Question about IT policies and procedures",
        "Request for IT training session",
        "Inquiry about upcoming system changes",
        "Need IT support for company event",
        "Question about IT asset management",
        "Request for IT department contact",
        "Inquiry about third-party integrations",
        "Need clarification on IT guidelines",
        "Question about data retention policy",
        "Request for IT capacity planning",
        "Inquiry about IT security certifications",
    ],
}

# Augmentation patterns to create variations
PREFIXES = [
    "", "Urgent: ", "Help needed: ", "Issue: ", "Problem: ", "Request: ",
    "Critical: ", "FYI: ", "Question: ", "Support needed: "
]

SUFFIXES = [
    "",
    " This is affecting my work.",
    " Please help as soon as possible.",
    " Been having this issue for days.",
    " Need this resolved urgently.",
    " This is blocking project deadline.",
    " Multiple users affected.",
    " Started happening after update.",
    " Tried restarting but still occurs.",
    " This is a recurring problem.",
]

def generate_ticket(category: str, template: str) -> dict:
    """Generate a single ticket with variations."""
    prefix = np.random.choice(PREFIXES)
    suffix = np.random.choice(SUFFIXES)
    
    # Create subject from template
    subject = template[:60] if len(template) > 60 else template
    
    # Create description with more detail
    description = f"{prefix}{template}{suffix}"
    
    # Add some additional context randomly
    if np.random.random() > 0.5:
        contexts = [
            " I've already tried basic troubleshooting steps.",
            " This started happening this morning.",
            " Other team members have the same issue.",
            " Works fine on my personal device.",
            " Was working yesterday, not today.",
        ]
        description += np.random.choice(contexts)
    
    return {
        "subject": subject,
        "description": description,
        "category": category
    }

def generate_dataset(samples_per_class: int = 400) -> pd.DataFrame:
    """Generate full dataset."""
    data = []
    
    for category, templates in CATEGORIES.items():
        for _ in range(samples_per_class):
            template = np.random.choice(templates)
            ticket = generate_ticket(category, template)
            data.append(ticket)
    
    return pd.DataFrame(data)

def main():
    print("Generating IT Service Desk Ticket Dataset...")
    
    # Generate data
    df = generate_dataset(samples_per_class=400)
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Total samples: {len(df)}")
    print(f"Categories: {df['category'].nunique()}")
    print("\nClass distribution:")
    print(df['category'].value_counts())
    
    # Create temporal-like split (80/10/10)
    train_df, temp_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['category']
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df['category']
    )
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_df)}")
    print(f"  Val: {len(val_df)}")
    print(f"  Test: {len(test_df)}")
    
    # Save to data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    train_df.to_csv(data_dir / "train.csv", index=False)
    val_df.to_csv(data_dir / "val.csv", index=False)
    test_df.to_csv(data_dir / "test.csv", index=False)
    
    print(f"\nFiles saved to {data_dir}/")
    print("  - train.csv")
    print("  - val.csv") 
    print("  - test.csv")

if __name__ == "__main__":
    main()
