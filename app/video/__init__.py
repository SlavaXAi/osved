"""On-demand video enrichment; collection itself never downloads transcripts."""

from app.video.transcripts import TranscriptResult, TranscriptSegment, TranscriptService

__all__ = ["TranscriptResult", "TranscriptSegment", "TranscriptService"]
