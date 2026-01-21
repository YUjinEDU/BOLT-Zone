# 유튜브 다운로드 빠른 시작 🎬

## 1️⃣ 설치

```bash
pip install yt-dlp
```

## 2️⃣ 단일 영상 다운로드

```bash
python scripts/download_youtube.py \
    --url "https://www.youtube.com/watch?v=VIDEO_ID" \
    --domain umpire \
    --video-id umpire_0001
```

## 3️⃣ Manifest 기반 일괄 다운로드 (권장)

### Step 1: Manifest 편집
`data/youtube_manifest.json` 파일을 열어서 실제 YouTube URL 입력:

```json
{
  "videos": [
    {
      "video_id": "umpire_0001",
      "url": "https://www.youtube.com/watch?v=실제_비디오_ID",
      "domain": "umpire",
      "outcome": "strike",
      "notes": "설명"
    }
  ]
}
```

### Step 2: 다운로드 실행
```bash
python scripts/download_youtube.py --manifest data/youtube_manifest.json
```

## 4️⃣ 결과 확인

- 영상: `data/raw/{domain}/{video_id}.mp4`
- 메타데이터: `data/metadata/youtube_downloads.json` (자동 생성)

## 📚 상세 가이드

전체 가이드는 [`docs/youtube_download_guide.md`](../docs/youtube_download_guide.md) 참조

## ⚠️ 저작권 주의

- 학습용: Fair use (비영리 연구) ✅
- 논문 메인 실험: 직접 촬영 권장
- 데이터셋 공개: 직접 촬영만 가능
