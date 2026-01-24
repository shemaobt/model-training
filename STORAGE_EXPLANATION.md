# Audio File Storage Explanation

## Current Setup (Modal Volumes)

**You don't need external cloud storage** - Modal volumes are already cloud storage!

### How it works:

1. **Upload Step** (`upload_audio_to_modal.py`):
   - Reads MP3 files from your local directories
   - Uploads them to Modal's persistent volume (`bible-audio-data`)
   - This volume is **cloud storage managed by Modal**

2. **Processing** (`modal_phase1.py`):
   - Processes files directly from Modal volume
   - Stores results back in the same volume
   - All data persists between runs

3. **Access**:
   - Files stay in Modal volume (persistent)
   - Accessible from any Modal function
   - No need to re-upload if you run processing again

### Modal Volume Benefits:
- ✅ Persistent storage (survives container restarts)
- ✅ Fast access from Modal functions
- ✅ No additional setup needed
- ✅ Automatic backups by Modal

## When You Might Want External Cloud Storage

You only need external cloud storage (S3, GCS, etc.) if you want to:

1. **Share files** with people/services outside Modal
2. **Backup** to a different provider
3. **Access from other services** (not Modal)
4. **Long-term archival** at lower cost

## Options

### Option 1: Use Modal Volumes Only (Recommended)
**Simplest approach** - just use what's already set up:
```bash
# Upload files
modal run upload_audio_to_modal.py

# Process
modal run modal_phase1.py
```

### Option 2: Add Cloud Storage Sync (Optional)
If you want to also store in S3/GCS, use `modal_phase1_with_cloud.py`:

1. **Set up credentials** in Modal:
   ```bash
   # For AWS S3
   modal secret create aws-credentials \
     AWS_ACCESS_KEY_ID=your_key \
     AWS_SECRET_ACCESS_KEY=your_secret
   
   # For Google Cloud Storage
   modal secret create gcs-credentials \
     GOOGLE_APPLICATION_CREDENTIALS_JSON='{"type":"service_account",...}'
   ```

2. **Run with cloud sync**:
   ```python
   import modal
   app = modal.App.lookup("bible-audio-training")
   
   # After processing, sync to S3
   app.sync_to_cloud_storage.remote(
       bucket_name="my-bible-audio-bucket",
       storage_type="s3"
   )
   ```

### Option 3: Download from Modal Volume
If you want to download results locally or to another service:

```python
import modal

app = modal.App.lookup("bible-audio-training")
volume = modal.Volume.from_name("bible-audio-data")

# Download specific files
# (You'd need to create a function for this)
```

## Recommendation

**Start with Modal volumes only** - they're already cloud storage and work perfectly for this use case. Only add external cloud storage if you have a specific need (sharing, backup, integration with other services).

## File Locations

- **Modal Volume**: `/mnt/audio_data/` (persistent cloud storage)
  - Raw MP3s: `/mnt/audio_data/raw_audio/`
  - Converted WAVs: `/mnt/audio_data/converted_audio/`
  - Results: `/mnt/audio_data/satere_units/`

- **Local** (for notebook): `~/bible_audio_project/`
  - Only used if running notebook locally
