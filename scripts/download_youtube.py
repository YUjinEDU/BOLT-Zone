"""
YouTube Video Downloader for BOLT-Zone Dataset

유튜브 영상을 최고 화질로 다운로드하고 메타데이터를 자동 기록합니다.

Requirements:
    pip install yt-dlp

Usage:
    # 단일 영상 다운로드
    python download_youtube.py --url "https://youtube.com/watch?v=..."
    
    # Manifest 파일 기반 일괄 다운로드
    python download_youtube.py --manifest data/youtube_manifest.json
    
    # 도메인 지정
    python download_youtube.py --url "..." --domain umpire
"""

import argparse
import json
import subprocess
from pathlib import Path
from datetime import datetime
import sys
import re


def download_video(url: str, output_dir: Path, video_id: str = None) -> dict:
    """
    유튜브 영상을 최고 화질로 다운로드
    
    Args:
        url: 유튜브 URL
        output_dir: 저장 디렉토리
        video_id: 사용자 정의 비디오 ID (선택)
    
    Returns:
        메타데이터 딕셔너리
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 영상 정보 먼저 가져오기 (메타데이터)
    print(f"📥 Fetching video info: {url}")
    info_cmd = [
        'yt-dlp',
        '--dump-json',
        '--no-playlist',
        url
    ]
    
    try:
        result = subprocess.run(info_cmd, capture_output=True, text=True, check=True)
        video_info = json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error fetching video info: {e.stderr}")
        raise
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing video info: {e}")
        raise
    
    # 2. 메타데이터 추출
    metadata = {
        'video_id': video_id or video_info.get('id'),
        'youtube_id': video_info.get('id'),
        'url': url,
        'title': video_info.get('title'),
        'channel': video_info.get('uploader'),
        'channel_id': video_info.get('channel_id'),
        'duration': video_info.get('duration'),  # seconds
        'upload_date': video_info.get('upload_date'),
        'resolution': f"{video_info.get('width')}x{video_info.get('height')}",
        'fps': video_info.get('fps'),
        'format_note': video_info.get('format_note'),
        'filesize': video_info.get('filesize'),
        'download_date': datetime.now().isoformat(),
    }
    
    # 3. 파일명 생성
    if video_id:
        filename = f"{video_id}.mp4"
    else:
        # 제목 기반 안전한 파일명
        safe_title = re.sub(r'[<>:"/\\|?*]', '_', video_info.get('title', 'video'))
        safe_title = safe_title[:50]  # 길이 제한
        filename = f"{video_info.get('id')}_{safe_title}.mp4"
    
    output_path = output_dir / filename
    
    # 4. 최고 화질로 다운로드
    print(f"⬇️  Downloading: {metadata['title']}")
    print(f"   Resolution: {metadata['resolution']} @ {metadata['fps']}fps")
    
    download_cmd = [
        'yt-dlp',
        '-f', 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',  # 최고 화질
        '--merge-output-format', 'mp4',
        '-o', str(output_path),
        '--no-playlist',
        url
    ]
    
    try:
        subprocess.run(download_cmd, check=True)
        print(f"✅ Downloaded: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Download failed: {e}")
        raise
    
    # 5. 실제 파일 정보 업데이트
    if output_path.exists():
        metadata['local_path'] = str(output_path.relative_to(Path.cwd()))
        metadata['filesize_actual'] = output_path.stat().st_size
    
    return metadata


def load_manifest(manifest_path: Path) -> list:
    """
    Manifest JSON 파일 로드
    
    Format:
    {
        "videos": [
            {
                "url": "https://youtube.com/watch?v=...",
                "video_id": "umpire_0001",
                "domain": "umpire",
                "outcome": "strike",
                "notes": "MLB Official - Fastball"
            }
        ]
    }
    """
    with open(manifest_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('videos', [])


def save_metadata(metadata_list: list, output_path: Path):
    """메타데이터를 JSON 파일로 저장"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'download_info': {
                'total_videos': len(metadata_list),
                'last_updated': datetime.now().isoformat()
            },
            'videos': metadata_list
        }, f, indent=2, ensure_ascii=False)
    
    print(f"📝 Metadata saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Download YouTube videos for BOLT-Zone dataset with metadata tracking'
    )
    parser.add_argument(
        '--url',
        type=str,
        help='YouTube URL to download'
    )
    parser.add_argument(
        '--manifest',
        type=Path,
        help='Path to manifest JSON file for batch download'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/raw'),
        help='Output directory (default: data/raw)'
    )
    parser.add_argument(
        '--domain',
        type=str,
        choices=['umpire', 'catcher_pov', 'custom'],
        help='Domain category (umpire, catcher_pov, custom)'
    )
    parser.add_argument(
        '--video-id',
        type=str,
        help='Custom video ID (default: auto-generated from YouTube ID)'
    )
    parser.add_argument(
        '--metadata-output',
        type=Path,
        default=Path('data/metadata/youtube_downloads.json'),
        help='Metadata output file (default: data/metadata/youtube_downloads.json)'
    )
    
    args = parser.parse_args()
    
    # Validation
    if not args.url and not args.manifest:
        parser.error('Either --url or --manifest must be provided')
    
    # yt-dlp 설치 확인
    try:
        subprocess.run(['yt-dlp', '--version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ yt-dlp is not installed!")
        print("Install it with: pip install yt-dlp")
        sys.exit(1)
    
    metadata_list = []
    
    # Single URL download
    if args.url:
        output_dir = args.output_dir
        if args.domain:
            output_dir = output_dir / args.domain
        
        try:
            metadata = download_video(args.url, output_dir, args.video_id)
            if args.domain:
                metadata['domain'] = args.domain
            metadata_list.append(metadata)
        except Exception as e:
            print(f"❌ Failed to download {args.url}: {e}")
            sys.exit(1)
    
    # Batch download from manifest
    elif args.manifest:
        videos = load_manifest(args.manifest)
        print(f"📋 Loading manifest: {len(videos)} videos")
        
        for i, video_entry in enumerate(videos, 1):
            print(f"\n[{i}/{len(videos)}] Processing...")
            url = video_entry.get('url')
            video_id = video_entry.get('video_id')
            domain = video_entry.get('domain', 'unknown')
            
            output_dir = args.output_dir / domain
            
            try:
                metadata = download_video(url, output_dir, video_id)
                # Manifest의 추가 정보 병합
                metadata.update({
                    'domain': domain,
                    'outcome': video_entry.get('outcome'),
                    'notes': video_entry.get('notes'),
                    'pitcher_handedness': video_entry.get('pitcher_handedness'),
                    'batter_handedness': video_entry.get('batter_handedness')
                })
                metadata_list.append(metadata)
                
            except Exception as e:
                print(f"⚠️  Failed to download {url}: {e}")
                print("   Continuing with next video...")
                continue
    
    # Save metadata
    if metadata_list:
        save_metadata(metadata_list, args.metadata_output)
        print(f"\n✅ Successfully downloaded {len(metadata_list)} video(s)")
    else:
        print("\n⚠️  No videos were downloaded")
        sys.exit(1)


if __name__ == '__main__':
    main()
