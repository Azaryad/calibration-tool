"""Light-based video/force calibration tool.

A clean rebuild of the swim-segment calibration pipeline. The core idea that
makes detection robust (and that the previous attempt missed):

  * The lamp is a WHITE DISC that saturates when on. The right signal is the
    *area of locally-bright pixels* in the ROI, measured relative to the
    surrounding ring -- not raw brightness (which sunlit caustics also reach).
  * The adaptive threshold is hard-clamped below the observed peak, so it can
    never exceed the maximum score (the bug that gave CamB/CamD zero events).
  * The CSV pulse pattern (two pulses per swim, distinct per-swim durations) is
    used as a matched-filter prior to pick the correct event pairs and reject
    caustic false positives.
"""

__all__ = ["ffmpeg_utils", "csv_utils", "detect"]
