# Performance Audit and Safe Optimization Plan — Sarvam Speech‑to‑Text (Tamil)
## Audio → segments.json Pipeline

**Date**: September 1, 2025  
**Pipeline Version**: Sarvam-adapted  
**Total Runtime (baseline)**: ~26 minutes for 5 audio files (original measurement)  
**Target**: Reduce end-to-end runtime without degrading output quality; replace local Whisper transcription with Sarvam speech‑to‑text (Saarika) for Tamil.

---

## Executive Summary

We will replace the local Whisper (large-v3) transcription stage with the **Sarvam** Speech→Text API using the **Saarika** model (best choice for Tamil transcription) and optionally **Saaras** for direct speech→English translation. This change eliminates local model loading and heavy CPU/GPU transcription time, moving the workload to a managed API service. Key wins:

- Remove local model cold-start time and GPU/CPU scaling complexity.  
- Potentially large wall‑time reduction for the transcription step due to vendor-managed infra and warm models.  
- Keep features you depend on (timestamps, diarization, code‑mix handling, language detection).

Tradeoffs:
- Introduces network latency and dependence on vendor availability/pricing.  
- Requires rate-limiting, retry/backoff, and transcript caching to avoid repeated costs.

---

## Updated Performance Hotspots (post Sarvam integration)

> **Notes:** The numbers below are conceptual — actual runtime gains must be measured via a short benchmark (see Benchmark Plan). The original pipeline bottlenecks were measured with local Whisper (19 minutes). After Sarvam integration, the local model loading bottleneck is eliminated, but network/API latency and per-file provider processing time become the new dominant factors.

### 1. **Sarvam ASR Transcription (Saarika)** ⭐⭐⭐⭐⭐ (PRIMARY)
- **Role**: Replace Whisper transcription with Sarvam batch API (model `saarika:v2.5`) configured for Tamil (`ta-IN`) or `auto` for detection.  
- **Capabilities**: Timestamps, speaker diarization, code‑mix handling, language detection, long-file batch jobs (up to 1h per file).  
- **New Bottlenecks**: Network transfer time for audio upload, API processing time, and per-minute billing/cost.

### 2. **Network / API Overhead** ⭐⭐⭐⭐ (HIGH)
- **Runtime Impact**: Upload + vendor processing time per file; concurrent uploads need careful rate limiting.  
- **Mitigations**: Use presigned URLs or direct HTTP file references where supported; chunked uploads; concurrency caps.

### 3. **Translation / Post-processing (optional)** ⭐⭐⭐ (MEDIUM)
- **Role**: If you need English translation, use `Saaras` or a translation step; keep translations batched & cached.  

### 4. **Other steps** (Audio conversion, NLU, Output generation) remain low impact but will be re-validated after switching ASR.

---

## Implementation Details — how to switch to Sarvam

Below are concrete changes to the pipeline, config, and code examples to integrate Sarvam safely and efficiently.

### Config changes (`config.yaml`)
```yaml
# Previously: asr_model: "large-v3"
asr_provider: "sarvam"
asr_model: "saarika:v2.5"  # Saarika for Tamil speech->text
asr_language_code: "ta-IN"  # or "auto" for detection
asr_batch_enabled: true
asr_batch_max_files_per_job: 20
asr_concurrency_limit: 3  # tune by testing
asr_retry: { max_attempts: 5, backoff_factor: 2 }
transcript_cache_enabled: true
transcript_cache_ttl_days: 30
```

### High level pipeline change
- Replace local `whisper.transcribe(file)` call with a `sarvam.transcribe_batch(files, model, options)` step.
- Keep a `transcript_cache` keyed by `sha256(audio_bytes) + model_name + language_code` to avoid duplicate API calls.
- Use asynchronous batch jobs for files >30s; use synchronous endpoint only for very short test clips.

### Recommended concurrency & reliability patterns
- Limit parallel uploads to `asr_concurrency_limit` (default 3).  
- Implement exponential backoff with jitter for API 429/5xx responses.  
- Use streaming download of outputs if the provider offers an output URL (less memory pressure).

---

## Code Examples

> The snippets below are scaffold‑level and include placeholders for your API key and file paths. Adapt to your repo style and error handling conventions.

### A — Simple curl (synchronous, short audio)
```bash
curl -X POST "https://api.sarvam.ai/speech-to-text" \
  -H "api-subscription-key: $SARVAM_KEY" \
  -F "file=@Audio1.wav" \
  -F "model=saarika:v2.5" \
  -F "language_code=ta-IN"
```

### B — Python batch flow (recommended for your longer files)
```python
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
import requests

SARVAM_KEY = "YOUR_API_KEY"
SARVAM_BATCH_CREATE = "https://api.sarvam.ai/v1/batch_jobs"
SARVAM_BATCH_UPLOAD = "https://api.sarvam.ai/v1/batch_uploads"  # provider might vary

# Simple transcript cache (in-memory or persistent)
TRANSCRIPT_CACHE = {}

def audio_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def create_batch_job(file_refs, language_code="ta-IN", model="saarika:v2.5"):
    payload = {
        "model": model,
        "language_code": language_code,
        "files": file_refs,
        "with_timestamps": True,
        "with_diarization": True
    }
    headers = {"api-subscription-key": SARVAM_KEY}
    r = requests.post(SARVAM_BATCH_CREATE, json=payload, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()


def upload_file_to_provider(local_path):
    # If Sarvam supports direct upload endpoints, use them. If not, upload then pass URL.
    files = {'file': open(local_path, 'rb')}
    headers = {"api-subscription-key": SARVAM_KEY}
    r = requests.post(SARVAM_BATCH_UPLOAD, files=files, headers=headers)
    r.raise_for_status()
    return r.json()['file_url']


def transcribe_files(file_paths):
    # 1) Check cache
    file_refs = []
    for p in file_paths:
        h = audio_hash(p)
        cache_key = f"{h}:saarika:v2.5:ta-IN"
        if cache_key in TRANSCRIPT_CACHE:
            file_refs.append({"path": p, "cached": True, "transcript": TRANSCRIPT_CACHE[cache_key]})
            continue
        # upload
        file_url = upload_file_to_provider(p)
        file_refs.append({"path": p, "cached": False, "file_url": file_url, "cache_key": cache_key})

    # 2) Create batch job with file URLs for the items that need transcription
    upload_urls = [r['file_url'] for r in file_refs if not r.get('cached')]
    if upload_urls:
        job = create_batch_job(upload_urls)
        job_id = job['id']

        # 3) Poll job status with backoff
        for attempt in range(60):
            status = requests.get(f"{SARVAM_BATCH_CREATE}/{job_id}", headers={"api-subscription-key": SARVAM_KEY}).json()
            if status['state'] in ("completed", "failed"):
                break
            time.sleep(min(2 ** attempt, 30))

        if status['state'] != 'completed':
            raise RuntimeError(f"Sarvam job failed or timed out: {status}")

        # 4) Download transcripts and populate cache
        for out in status['outputs']:
            cache_key = out['meta']['cache_key'] if 'meta' in out else None
            transcript = out['transcript']
            # save to cache and to local storage
            if cache_key:
                TRANSCRIPT_CACHE[cache_key] = transcript

    # 5) Merge cached + new transcripts and return structured segments
    results = []
    for r in file_refs:
        if r.get('cached'):
            results.append({"path": r['path'], "transcript": r['transcript']})
        else:
            # map outputs by file_url or path - implementation specific
            results.append({"path": r['path'], "transcript": '...retrieved from job outputs...'})

    return results
```

---

## Optimization Recommendations (Sarvam-specific)

### Priority A — Fast wins (expected improvement vs local Whisper)
1. **Switch to Sarvam batch API** for all files >30s to remove local transcription CPU overhead.  
2. **Transcript cache** keyed by audio hash + model to avoid re-transcribing the same audio.  
3. **Use batch (async) job for long files** and synchronous only for very short test clips.

### Priority B — Reliability & cost control
1. **Limit concurrency** (default 3) and implement exponential backoff with jitter for 429/5xx.  
2. **Use server-side upload URLs** or presigned URLs if supported to speed uploads and reduce memory use.  
3. **Log cost metadata** (duration, cost estimate per file) to monitor spend.

### Priority C — Optional quality-focused steps
1. **Try `language_code=auto`** on some files to check auto detection vs explicit `ta-IN`.  
2. **If translation required**, experiment with `saaras` for speech→English to avoid a separate translate step.  
3. **Compare vendor diarization to local NLU heuristics** — you may be able to drop expensive local diarization if vendor output meets quality needs.

---

## Benchmark Plan (run this before full rollout)
1. **Files**: Use the same 3 representative files (short, medium, long) you already have: Audio3.wav (~0:47), Audio1.wav (~3:18), Audio5.wav (~8:50).  
2. **Measurements** per provider (Sarvam + current Whisper baseline):
   - Wall-clock end-to-end time (upload + processing + post)  
   - Cost estimate (provider minutes * price/min)  
   - WER vs ground truth (use a small labeled set)  
   - Diarization quality (if speaker labels matter)  
   - Failure rate and retry counts
3. **Run**: Transcribe each file 5x to capture variance.  
4. **Report**: Produce a table (Latency | Cost | WER | Diarization) and decide thresholds for rollout.

---

## Changes to Prior Optimization Recommendations

The original plan assumed optimizing a local Whisper model — many of those steps are still relevant but will be re-scoped:

- **Model size reduction / GPU caching / lazy loading**: No longer applicable for the ASR step because Sarvam is API-managed. Remove local model size tuning from the main ASR pipeline. Keep GPU plans only if you retain any local models (e.g., for fallback or advanced offline use).  
- **Batch processing**: Keep the batch-processing concept, but implement at the API level (group uploads into a single Sarvam job when supported).  
- **Model caching** (internal cache of model object) → replaced by **transcript caching** (avoid re‑calls to Sarvam for the same audio).  

Other pipeline optimization items (translation batching, NLU regex compilation, streaming JSON output, skipping conversion) remain fully relevant.

---

## Implementation Plan (Phased)

### Phase 0 — Safety & preparation (Day 0)
- Acquire Sarvam API key and test account.  
- Set up secure secrets storage for `SARVAM_KEY`.  
- Add config knobs `asr_provider` and `asr_concurrency_limit`.

### Phase 1 — Replace transcription (Days 1–3)
- Implement `sarvam_transcribe_batch` step with transcript cache.  
- Add robust retry/backoff and concurrency limiting.  
- Run benchmark plan.  
- Validate outputs (WER, speaker labels, timestamps) vs baseline.

### Phase 2 — Tune & harden (Days 4–7)
- Tune concurrency and chunking based on benchmark results.  
- Enable `saaras` translation experiment if needed.  
- Add cost logging and alerting thresholds.

### Phase 3 — Rollout & monitoring (Week 2)
- Gradual rollout: route 10% of jobs to Sarvam, compare automatically, then 50% and full.  
- Add monitoring: latency histograms, error rate, cost per minute, WER spot checks.

---

## Quality Preservation Measures

All prior quality controls remain. In addition:
- **Transcript cache + E2E A/B testing**: run a small A/B test between Sarvam outputs and Whisper baseline on random sample of files.  
- **Automated WER sampling**: compute WER for a labeled holdout set weekly.  
- **Diarization vs local NLU tests**: ensure speaker labels meet the downstream NLU expectations (if not, retain local heuristics as fallback).  

---

## Risk Assessment (Sarvam integration)

### Low Risk
- ✅ Skipping WAV conversion
- ✅ NLU pattern compilation
- ✅ Streaming JSON output

### Medium Risk
- ⚠️ Transcript cache correctness (stale cache) — mitigate with TTL and cache invalidation.  
- ⚠️ Network transient errors — mitigate with retries and alerting.

### High Risk
- 🔴 Vendor outage or significant pricing changes — keep a local Whisper fallback path (smaller model) for crucial/time-sensitive workloads.
- 🔴 Data‑privacy / regulatory constraints — ensure Sarvam data retention policy fits compliance; if not, keep local/on‑prem solution.

---

## Success Metrics (updated)

### Primary Metrics
- **Runtime reduction**: measurable improvement in transcription wall time vs baseline (target: substantial reduction; exact % TBD after benchmark).  
- **Quality preservation**: WER change < 5% on the labeled validation set.  
- **Cost awareness**: cost per minute tracked and within budget thresholds.

### Secondary Metrics
- **Error rate**: <1% job failures after retries  
- **Scalability**: ability to sustain configured concurrency without throttling  
- **Rollback capability**: simple toggle to route jobs back to local Whisper fallback

---

## Rollback & Fallback Strategy
1. Keep the local Whisper transcription path available (smaller `medium` model recommended) as an emergency fallback (toggle via `asr_provider`).  
2. Implement a health-checker for the Sarvam API; on repeated failures automatically switch to local fallback for new jobs and alert ops.  

---

## Next Steps (suggested immediate actions)
1. Provision Sarvam API credentials and test the synchronous endpoint with a short Tamil clip.  
2. Implement the transcript cache and batch job scaffolding in a feature branch.  
3. Run the 3-file benchmark comparing Sarvam vs local Whisper baseline and produce the latency | cost | WER table.  

---

*Document prepared to replace Whisper transcription with Sarvam (Saarika) for Tamil audio. Implementations should use the batch job API for longer recordings and keep robust caching, rate-limiting, and fallback options.*

