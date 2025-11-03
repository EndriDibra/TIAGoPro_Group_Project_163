#!/usr/bin/env python3
"""
Aria2c-based Rosbag Downloader for SCAND Dataset
Uses aria2c for optimized parallel downloads with multiple connections per file.
"""

import os
import subprocess
import requests
import tempfile
from pathlib import Path

# --- Configuration ---
BASE_URL = 'https://dataverse.tdl.org/'
DOI = "doi:10.18738/T8/0PRYRH"
DOWNLOAD_DIR = "miniproject/rosbags"

# Aria2c settings
CONNECTIONS_PER_FILE = 16  # Parallel connections per file
MAX_CONCURRENT_FILES = 8   # Files to download simultaneously
MIN_SPLIT_SIZE = "1M"      # Minimum size to split downloads

# Create download directory
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def check_aria2c():
    """Check if aria2c is installed."""
    try:
        subprocess.run(['aria2c', '--version'], 
                      capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def main():
    print("="*80)
    print("SCAND Dataset - Aria2c Downloader".center(80))
    print("="*80)
    
    # Check for aria2c
    if not check_aria2c():
        print("\n❌ Error: aria2c is not installed!")
        print("\nInstall with:")
        print("   sudo apt-get install aria2")
        print("\nOr:")
        print("   sudo snap install aria2c")
        return
    
    # Get dataset metadata
    print("\n⟳ Fetching dataset information...")
    metadata_url = f"{BASE_URL}api/datasets/:persistentId/?persistentId={DOI}"
    response = requests.get(metadata_url)
    response.raise_for_status()
    dataset = response.json()
    
    # Filter for rosbag files
    bag_files = []
    for f in dataset['data']['latestVersion']['files']:
        filename = f["dataFile"]["filename"]
        if filename.endswith(".bag"):
            bag_files.append({
                'filename': filename,
                'file_id': f["dataFile"]["id"],
                'size': f["dataFile"].get("filesize", 0)
            })
    
    total_size_gb = sum(f['size'] for f in bag_files) / (1024**3)
    
    print(f"\n📦 Dataset Summary:")
    print(f"   • Total files: {len(bag_files)} rosbag files")
    print(f"   • Total size: {total_size_gb:.2f} GB")
    print(f"   • Connections per file: {CONNECTIONS_PER_FILE}")
    print(f"   • Concurrent downloads: {MAX_CONCURRENT_FILES}")
    print(f"   • Download directory: {DOWNLOAD_DIR}/")
    
    # Create input file for aria2c
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        input_file = f.name
        for file_info in bag_files:
            download_url = f"{BASE_URL}api/access/datafile/{file_info['file_id']}"
            f.write(f"{download_url}\n")
            f.write(f"  out={file_info['filename']}\n")
    
    print(f"\n{'='*80}\n")
    print("🚀 Starting aria2c download...\n")
    
    # Run aria2c
    aria2c_cmd = [
        'aria2c',
        f'--input-file={input_file}',
        f'--dir={DOWNLOAD_DIR}',
        f'--max-connection-per-server={CONNECTIONS_PER_FILE}',
        f'--max-concurrent-downloads={MAX_CONCURRENT_FILES}',
        f'--min-split-size={MIN_SPLIT_SIZE}',
        '--split=16',                    # Split file into 16 parts
        '--continue=true',               # Resume downloads
        '--max-tries=5',                 # Retry failed downloads
        '--retry-wait=3',                # Wait 3s between retries
        '--timeout=60',                  # Connection timeout
        '--connect-timeout=30',          # Connect timeout
        '--summary-interval=10',         # Progress every 10s
        '--console-log-level=notice',    # Cleaner output
        '--human-readable=true',         # Human-readable sizes
        '--file-allocation=none',        # Faster on some systems
    ]
    
    try:
        result = subprocess.run(aria2c_cmd, check=False)
        
        # Clean up temp file
        os.unlink(input_file)
        
        print("\n" + "="*80)
        if result.returncode == 0:
            print("✓ Download Complete!".center(80))
        else:
            print("⚠ Download completed with some errors".center(80))
        print("="*80)
        
        # Verify downloaded files
        print(f"\n📊 Verification:")
        downloaded = 0
        total_downloaded_size = 0
        
        for file_info in bag_files:
            filepath = Path(DOWNLOAD_DIR) / file_info['filename']
            if filepath.exists():
                downloaded += 1
                total_downloaded_size += filepath.stat().st_size
        
        print(f"   • Downloaded: {downloaded}/{len(bag_files)} files")
        print(f"   • Total size: {total_downloaded_size/(1024**3):.2f} GB")
        
        if downloaded < len(bag_files):
            print(f"\n⚠ Warning: {len(bag_files) - downloaded} file(s) missing")
            print("   Run the script again to resume incomplete downloads.")
        
        print("="*80)
        
    except KeyboardInterrupt:
        print("\n\n⚠ Download interrupted by user")
        os.unlink(input_file)
        print("   Run the script again to resume.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        os.unlink(input_file)

if __name__ == "__main__":
    main()